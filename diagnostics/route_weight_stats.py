"""Read-only routing-weight health check for --route_fix runs (A 方案配套).

Verdict context (2026-10-27): full-pool soft pull caused cold-start key
collapse (round-0 effective rank 87.8 -> 6.5, keys collapsed to ~6
directions => near-identical prompt mixture for every sample). The pull
term is now decoupled (official top-k + batchwise-majority slots,
gradient only through selected keys), so this script monitors whether
routing stays informative on the NEW run.

Zero training impact: consumes a saved round checkpoint, rebuilds the
outer partition from its manifest (no RNG replay), extracts frozen-ViT
cls features once, and reports four pre-registered metrics plus the key
effective rank:

  1. mean routing entropy      H(w) bits per sample (soft weights)
  2. mean max routing weight   E_b max_s w[b, s]
  3. top-1 slot usage          argmax(w) histogram entropy + top1 share
  4. between-sample diversity  mean pairwise cosine of w rows
                               (1.0 = identical routing for all samples
                                = degenerate prompt mixture)
  5. key effective rank        participation ratio of centered key matrix

Advisory thresholds (printed, not enforced): diversity < 0.5 and usage
entropy > 3 bits => routing discriminative; diversity > 0.9 => collapsed.

Usage:
  python diagnostics/route_weight_stats.py \
      --run_dir output/cifar100/fedta/cifar_route_fix/seed_42 \
      --round 0 --device cuda
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # diag_utils 同目录导入

from diagnostics.analyze_prompt_signal import (  # noqa: E402
    build_models, partition_from_manifest, l2_normalize)


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
    k = l2_normalize(key.float(), dim=1)
    centered = k - k.mean(0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    lam = s ** 2
    return float((lam.sum() ** 2 / (lam ** 2).sum()))


def routing_stats(w):
    """w: N, pool soft routing weights (rows sum to 1)."""
    n = w.shape[0]
    ent = -(w * torch.log2(w.clamp_min(1e-12))).sum(dim=1)
    max_w = w.max(dim=1).values
    argmax_counts = torch.bincount(w.argmax(dim=1), minlength=w.shape[1])
    freq = argmax_counts.double() / n
    nz = freq[freq > 0]
    usage_ent = float(-(nz * torch.log2(nz)).sum()) if nz.numel() else 0.0
    # between-sample diversity on a capped subsample
    sub = w
    if n > 512:
        idx = torch.randperm(n, device=w.device)[:512]
        sub = w[idx]
    sub_n = l2_normalize(sub, dim=1)
    cos = sub_n @ sub_n.t()
    off = cos[~torch.eye(sub_n.shape[0], dtype=torch.bool, device=w.device)]
    return {
        "n_samples": int(n),
        "mean_routing_entropy_bits": round(float(ent.mean()), 4),
        "mean_max_weight": round(float(max_w.mean()), 4),
        "usage_entropy_bits": round(usage_ent, 4),
        "top1_slot_share": round(float(freq.max()), 4),
        "between_sample_w_cosine": round(float(off.mean()), 4),
    }


def collect_w(original_model, subset, key_norm, tau, device, batch_size, cap):
    ws = []
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    seen = 0
    for x, _target in loader:
        if seen >= cap:
            break
        x = x.to(device)
        with torch.no_grad():
            cls = original_model(x)["pre_logits"]
        sim = l2_normalize(cls, dim=1) @ key_norm.t()
        ws.append(torch.softmax(sim / tau, dim=-1))
        seen += int(x.shape[0])
    return torch.cat(ws, dim=0)


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
    if not getattr(run_args, "route_fix", False):
        print("WARNING: checkpoint args route_fix=False — soft-routing "
              "stats are only meaningful for --route_fix runs")

    key_norm = l2_normalize(key, dim=1).to(device)
    eff_rank = key_effective_rank(key)

    original_model, _model = build_models(run_args, device)
    original_model.eval()
    client_data, _mask = partition_from_manifest(run_args,
                                                 raw["outer_split_manifest"])

    per_client = {}
    all_w = []
    for cid in range(run_args.client_num):
        per_task = {}
        for t in range(run_args.task_num):
            subset = client_data[cid][t]
            if len(subset) == 0:
                continue
            w = collect_w(original_model, subset, key_norm, tau, device,
                          args.batch_size, args.max_samples_per_client_task)
            per_task[str(t)] = routing_stats(w)
            all_w.append(w)
        per_client[str(cid)] = per_task

    w_all = torch.cat(all_w, dim=0)
    aggregate = routing_stats(w_all)

    diversity = aggregate["between_sample_w_cosine"]
    usage_bits = aggregate["usage_entropy_bits"]
    advisory = ("DEGENERATE (routing identical across samples)"
                if diversity > 0.9 else
                "PASS (discriminative)" if diversity < 0.5 and usage_bits > 3
                else "BORDERLINE")

    result = {
        "checkpoint": ckpt_path,
        "tau": tau,
        "key_effective_rank": round(eff_rank, 2),
        "key_pool_size": int(key.shape[0]),
        "aggregate": aggregate,
        "advisory": advisory,
        "per_client": per_client,
        "metric_definitions": {
            "mean_routing_entropy_bits": "H(w) averaged over samples",
            "mean_max_weight": "E_b max_s w[b,s]",
            "usage_entropy_bits": "entropy of argmax(w) slot histogram",
            "between_sample_w_cosine": "mean pairwise cosine of w rows "
                                       "(1.0 = same mixture for all samples)",
            "key_effective_rank": "participation ratio of centered "
                                  "normalized key matrix (100 = healthy)",
        },
    }
    print(json.dumps({k: result[k] for k in
                      ("checkpoint", "tau", "key_effective_rank", "aggregate",
                       "advisory")}, indent=2))
    out = args.out or os.path.join(
        args.run_dir, "route_weight_stats_%s.json" % args.round)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
