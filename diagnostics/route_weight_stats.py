"""Read-only routing-weight health check for --route_fix runs (A-plan gate).

Verdict context (2026-10-27): the first full-pool soft pull
(reduce_sim = top_k * sum(w*sim) / B) co-occurred with round-0 key
collapse (effective rank 87.8 -> 6.5, near-identical prompt mixture for
every sample) and is treated as a strong candidate collapse driver
pending real-training counterfactual. The pull term is now decoupled
(official top-k + batchwise-majority slots, gradient only through
selected keys), so this script gates the NEW run.

Caliber (per external review 2026-10-27, do not overstate):
  - This is a DISTRIBUTION-LEVEL health check, NOT an exact replay of
    training-time forward inputs: the outer client-task partition is
    rebuilt from the checkpoint manifest, while the inner-split and
    transform RNG are NOT restored — features come from freshly sampled
    random augmentations of the same distribution.
  - Only embedding_key='cls' is supported (checked, fail-closed): the
    script feeds original_model(x)["pre_logits"] as the routing query,
    matching Global_Prompt.forward's cls branch.
  - route_fix=False checkpoints get verdict NOT_APPLICABLE (gate not
    evaluated, exit code 2): official hard routing does not consume
    these softmax weights, so a PASS there would be meaningless. Such
    runs are only useful as a pipeline smoke test.

Gate criteria (pre-registered; evaluated ONLY when route_fix=True;
evaluated simultaneously on the global aggregate AND on every client's
pooled aggregate, so global aggregation cannot mask per-client
collapse):
  - between-sample w cosine < 0.5    (global AND each client)
  - argmax usage entropy   > 3 bits  (global AND each client)
  - key effective rank     > 50      (global; the key pool is shared)
Verdict DEGENERATE (exit 1) if any global/client diversity > 0.9.
Per client-task diversity > 0.9 is reported as an informational reason
(small per-task samples make it noisy) but does not flip the gate.
Exit codes: 0 = PASS, 1 = FAIL / DEGENERATE, 2 = NOT_APPLICABLE.

Usage:
  python diagnostics/route_weight_stats.py \
      --run_dir output/cifar100/fedta/cifar_route_fix/seed_42 \
      --round 0 --device cuda --data_path <DATA_PATH>
"""
import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # diag_utils 同目录导入

from diagnostics.analyze_prompt_signal import (  # noqa: E402
    build_models, partition_from_manifest, l2_normalize)

# 预注册门控阈值（RUNCOMMANDS.md §20 同步）
GATE_DIVERSITY_MAX = 0.5
GATE_USAGE_BITS_MIN = 3.0
GATE_EFF_RANK_MIN = 50.0
DEGENERATE_DIVERSITY = 0.9
SUBSAMPLE_MAX = 512
SUBSAMPLE_SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True,
                   help="run directory containing checkpoints/ and logs/")
    p.add_argument("--round", default="latest",
                   help="round number ('0' -> round_0000.pth) or 'latest'")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples_per_client_task", type=int, default=1024,
                   help="cap features per (client, task) for speed")
    p.add_argument("--out", default=None, help="JSON output path")
    return p.parse_args()


def load_checkpoint(run_dir, round_tag):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if round_tag == "latest":
        path = os.path.join(ckpt_dir, "latest.pth")
    else:
        path = os.path.join(ckpt_dir, "round_%04d.pth" % int(round_tag))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu"), path


def key_effective_rank(key):
    """Participation ratio of the centered normalized key matrix.

    Degenerate protection (per external review): zero/non-finite
    denominator returns 0.0 (fail-closed) instead of NaN.
    """
    k = l2_normalize(key.float(), dim=1)
    centered = k - k.mean(0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    lam = s ** 2
    num = lam.sum() ** 2
    den = (lam ** 2).sum()
    if den == 0 or not torch.isfinite(num) or not torch.isfinite(den):
        return 0.0
    ratio = float(num / den)
    return ratio if math.isfinite(ratio) else 0.0


def routing_stats(w):
    """w: N, pool soft routing weights (rows sum to 1).

    between_sample_w_cosine is None when fewer than 2 samples are
    available (undefined; the gate treats None as fail-closed).
    """
    n = w.shape[0]
    ent = -(w * torch.log2(w.clamp_min(1e-12))).sum(dim=1)
    max_w = w.max(dim=1).values
    argmax_counts = torch.bincount(w.argmax(dim=1), minlength=w.shape[1])
    freq = argmax_counts.double() / n
    nz = freq[freq > 0]
    usage_ent = float(-(nz * torch.log2(nz)).sum()) if nz.numel() else 0.0
    # between-sample diversity on a fixed-seed capped subsample
    # (deterministic across reruns of the same checkpoint)
    sub = w
    if n > SUBSAMPLE_MAX:
        g = torch.Generator().manual_seed(SUBSAMPLE_SEED)
        sub = w[torch.randperm(n, generator=g)[:SUBSAMPLE_MAX].to(w.device)]
    if sub.shape[0] < 2:
        diversity = None
    else:
        sub_n = l2_normalize(sub, dim=1)
        cos = sub_n @ sub_n.t()
        off = cos[~torch.eye(sub_n.shape[0], dtype=torch.bool,
                             device=w.device)]
        diversity = round(float(off.mean()), 4)
    return {
        "n_samples": int(n),
        "mean_routing_entropy_bits": round(float(ent.mean()), 4),
        "mean_max_weight": round(float(max_w.mean()), 4),
        "usage_entropy_bits": round(usage_ent, 4),
        "top1_slot_share": round(float(freq.max()), 4),
        "between_sample_w_cosine": diversity,
    }


def collect_w(original_model, subset, key_norm, tau, device, batch_size, cap):
    """Soft routing weights for up to exactly `cap` samples (truncated)."""
    ws = []
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    seen = 0
    for x, _target in loader:
        if seen >= cap:
            break
        x = x[:cap - seen].to(device)  # 精确截断，不越过 cap
        with torch.no_grad():
            cls = original_model(x)["pre_logits"]
        sim = l2_normalize(cls, dim=1) @ key_norm.t()
        ws.append(torch.softmax(sim / tau, dim=-1))
        seen += int(x.shape[0])
    return torch.cat(ws, dim=0)


def unit_gate(stats, label, reasons):
    """diversity < 0.5 AND usage entropy > 3 bits (fail-closed on None)."""
    div = stats["between_sample_w_cosine"]
    if div is None:
        reasons.append("%s: between-sample cosine undefined (<2 samples)"
                       % label)
        return False
    ok = True
    if not div < GATE_DIVERSITY_MAX:
        reasons.append("%s: between-sample w cosine %.4f >= %.2f"
                       % (label, div, GATE_DIVERSITY_MAX))
        ok = False
    if not stats["usage_entropy_bits"] > GATE_USAGE_BITS_MIN:
        reasons.append("%s: usage entropy %.4f bits <= %.1f"
                       % (label, stats["usage_entropy_bits"],
                          GATE_USAGE_BITS_MIN))
        ok = False
    return ok


def main():
    args = parse_args()
    raw, ckpt_path = load_checkpoint(args.run_dir, args.round)
    run_args = argparse.Namespace(**raw["args"])
    run_args.data_path = args.data_path
    device = args.device

    server_prompt = raw.get("server_prompt")
    if not server_prompt or "prompt_key" not in server_prompt:
        raise RuntimeError("checkpoint lacks server_prompt.prompt_key")
    key = server_prompt["prompt_key"].float()
    tau = float(getattr(run_args, "route_tau", 0.1))
    route_fix = bool(getattr(run_args, "route_fix", False))

    # 口径校验（fail-closed）：本脚本仅复算 embedding_key='cls' 的前向
    if getattr(run_args, "embedding_key", "cls") != "cls":
        raise RuntimeError(
            "route_weight_stats only supports embedding_key='cls' "
            "(checkpoint has %r)"
            % getattr(run_args, "embedding_key", None))

    # 先搬到目标设备再归一化，与训练前向同设备口径
    key_norm = l2_normalize(key.to(device), dim=1)
    eff_rank = key_effective_rank(key)

    original_model, _model = build_models(run_args, device)
    original_model.eval()
    client_data, _mask = partition_from_manifest(run_args,
                                                 raw["outer_split_manifest"])

    per_client = {}
    all_w = []
    for cid in range(run_args.client_num):
        per_task = {}
        client_ws = []
        for t in range(run_args.task_num):
            subset = client_data[cid][t]
            if len(subset) == 0:
                continue
            w = collect_w(original_model, subset, key_norm, tau, device,
                          args.batch_size, args.max_samples_per_client_task)
            per_task[str(t)] = routing_stats(w)
            client_ws.append(w)
            all_w.append(w)
        per_client[str(cid)] = {
            "tasks": per_task,
            "client_aggregate": (routing_stats(torch.cat(client_ws, dim=0))
                                 if client_ws else None),
        }

    w_all = torch.cat(all_w, dim=0)
    aggregate = routing_stats(w_all)

    reasons = []
    if not route_fix:
        verdict = "NOT_APPLICABLE"
        gate_pass = None
        reasons.append("checkpoint args route_fix=False: official hard "
                       "routing does not consume softmax weights; "
                       "gate not evaluated (pipeline smoke only)")
        for cid in range(run_args.client_num):
            per_client[str(cid)]["gate_pass"] = None
    else:
        # DEGENERATE 硬判：任一全局/客户端聚合 diversity > 0.9
        degenerate_units = []
        agg_div = aggregate["between_sample_w_cosine"]
        if agg_div is not None and agg_div > DEGENERATE_DIVERSITY:
            degenerate_units.append("global")
        for cid in range(run_args.client_num):
            st = per_client[str(cid)]["client_aggregate"]
            if st is not None and st["between_sample_w_cosine"] is not None \
                    and st["between_sample_w_cosine"] > DEGENERATE_DIVERSITY:
                degenerate_units.append("client %d" % cid)
        # client-task 级 DEGENERATE 仅作信息性提示（小样本噪声，不翻门控）
        for cid in range(run_args.client_num):
            for t, st in per_client[str(cid)]["tasks"].items():
                div = st["between_sample_w_cosine"]
                if div is not None and div > DEGENERATE_DIVERSITY:
                    reasons.append("INFO client %s task %s: diversity %.4f "
                                   "> %.1f (informational, not gated)"
                                   % (cid, t, div, DEGENERATE_DIVERSITY))
        if degenerate_units:
            verdict = "DEGENERATE"
            gate_pass = False
            reasons.append("routing identical across samples: %s"
                           % ", ".join(degenerate_units))
        else:
            # 全局三条件（key 池全局共享，有效秩只在全局判一次）
            ok = unit_gate(aggregate, "global", reasons)
            if not eff_rank > GATE_EFF_RANK_MIN:
                reasons.append("key effective rank %.2f <= %.0f"
                               % (eff_rank, GATE_EFF_RANK_MIN))
                ok = False
            # 每客户端门控：防全局聚合掩盖单客户端塌缩
            for cid in range(run_args.client_num):
                st = per_client[str(cid)]["client_aggregate"]
                if st is None:
                    reasons.append("client %d: no evaluable samples" % cid)
                    per_client[str(cid)]["gate_pass"] = False
                    ok = False
                    continue
                c_ok = unit_gate(st, "client %d" % cid, reasons)
                per_client[str(cid)]["gate_pass"] = c_ok
                ok = ok and c_ok
            gate_pass = ok
            verdict = "PASS" if ok else "FAIL"

    result = {
        "checkpoint": ckpt_path,
        "tau": tau,
        "route_fix": route_fix,
        "key_effective_rank": round(eff_rank, 2),
        "key_pool_size": int(key.shape[0]),
        "gate_criteria": {
            "between_sample_w_cosine_max": GATE_DIVERSITY_MAX,
            "usage_entropy_bits_min": GATE_USAGE_BITS_MIN,
            "key_effective_rank_min": GATE_EFF_RANK_MIN,
            "degenerate_diversity": DEGENERATE_DIVERSITY,
            "scope": "global aggregate AND every client pooled aggregate "
                     "(client-task DEGENERATE is informational only)",
        },
        "verdict": verdict,
        "gate_pass": gate_pass,
        "failure_reasons": reasons,
        "aggregate": aggregate,
        "per_client": per_client,
        "metric_definitions": {
            "mean_routing_entropy_bits": "H(w) averaged over samples",
            "mean_max_weight": "E_b max_s w[b,s]",
            "usage_entropy_bits": "entropy of argmax(w) slot histogram",
            "between_sample_w_cosine": "mean pairwise cosine of w rows "
                                       "(1.0 = same mixture for all samples; "
                                       "null = undefined, <2 samples)",
            "key_effective_rank": "participation ratio of centered "
                                  "normalized key matrix (theoretical max "
                                  "is pool_size-1 after centering; "
                                  ">50 = gate-healthy)",
        },
        "caliber_notes": "distribution-level check: manifest-rebuilt outer "
                         "client-task partition with freshly sampled random "
                         "transforms; NOT an exact replay of training-time "
                         "forward inputs (inner-split/transform RNG not "
                         "restored); embedding_key='cls' enforced",
    }
    print(json.dumps({k: result[k] for k in
                      ("checkpoint", "tau", "route_fix",
                       "key_effective_rank", "verdict", "gate_pass",
                       "aggregate")}, indent=2))
    if reasons:
        print("notes:")
        for r in reasons:
            print("  -", r)
    out = args.out or os.path.join(
        args.run_dir, "route_weight_stats_%s.json" % args.round)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved ->", out)

    if verdict == "PASS":
        sys.exit(0)
    if verdict == "NOT_APPLICABLE":
        sys.exit(2)
    sys.exit(1)


if __name__ == "__main__":
    main()
