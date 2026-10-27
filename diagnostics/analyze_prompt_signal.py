"""JE-premise probe: per-client prompt ablation on a finished FedTA run.

JE (部署损失反传 prompt) 的立项前提（INR_PHASE1_HEAD_DIVERGENCE.md §7/§8）:

    "训练结束后，多数客户端的 prompt 组件对分类的贡献接近零（噪声级）"

已有证据是梯度侧的（phase-1 masked CE 对 4/5 客户端退化为 ln(8) 均匀头，
G1 近噪声；部署 Chead 从不反传 prompt）。本脚本把该前提升级为部署准确率
证据：对已完成 run 的 round-end checkpoint 做纯推理 prompt 消融。

分析语义（对每个任务边界 after_task t、每个客户端 c）：

  1. baseline        : Client_DF.evaluate 原样复刻部署路径
                       original_model -> vit(prompt) -> Tail_Anchor -> 任务 Chead，
                       class_mask 对非任务类 -inf 掩码后 argmax；inner test 集
                       （data_split_state 的 30% 切分）, batch=8 shuffle=True
                       （主线 eval 默认）。每个 (client, task) 固定
                       torch.manual_seed(9700+100*cid+task)（仓库诊断脚本通例），
                       baseline 与各消融共享同一 batch 顺序与 dropout mask
                       （官方 evaluate 不切 eval 模式，Chead Dropout(0.5) 激活），
                       因此 Δ 只归因于消融本身。
  2. local zero      : 该 client 部署 prompt 的 prompt 池权重置零。
                       prompt_key 保留 -> key 匹配/路由不变，度量的是
                       prompt 内容对分类的因果贡献。
  3. local random    : prompt 池替换回初始化分布 U(-1,1)（Global_Prompt
                       prompt_init='uniform' 的原始分布），key 同样保留。
  4. G_prompt zero   : 消融 checkpoint 的 server_prompt（SIKF 融合后的全局
                       prompt）。注意 FedTA 的 round-end checkpoint 在广播之后
                       保存，所有 client 的 prompts 与 server_prompt 是同一组
                       权重（逐客户端 local 池只存在于轮内 phase-1 之后、融合
                       之前，从不落盘）——脚本运行时逐字节验证；若一致，
                       G_prompt 置零与 local 置零是同一操作，结果直接复用并
                       以 same_weights_as_local_zero 标注，不重复计算；
                       仅当不一致（假想的 mid-round 状态）才单独重测。
  5. key stats       : client 的 inner test 集过 prompt key 匹配（与
                       Global_Prompt.forward 完全相同的 l2 归一 + 余弦相似度，
                       逐样本 top-k，不含 batchwise_prompt 批内投票），统计
                       被选槽位频率分布的熵 / 不同槽位数 / top1 槽位占比 /
                       top1 相似度与 top1-top2 间隔——检验 "prompt 池是否
                       实际被差异化使用"。

数据重建（不重放任何随机流）：外层联邦划分直接由 checkpoint 的
outer_split_manifest（每 client-task 的原始样本索引 + class mask）构建，
inner 70/30 eval 集由 data_split_state 重建（Client_DF.rebuild_data_from_
indices）；全程不调用 spliter.random_split / torch random_split。消融全部
在内存中进行（load_prompts 生成 deepcopy 后改 .data），checkpoint 文件
只读、绝不写回。

已知边界（如实报告）：Server_DF.load_checkpoint 与 skip_fed_avg_head 的
checkpoint 不兼容（它把 global_head 模板重建为 clients[0].vit.head 的
Linear，而 HS-N run 的 global_head 保持 __init__ 的 Chead，键集不匹配），
因此本脚本按 client state 逐项恢复、绕开该 server 级加载路径；global_head
本身是部署路径不经过的死组件（INR_PHASE1_HEAD_DIVERGENCE.md §4.5），
不影响任何分析项。

Usage (repo root):
  python diagnostics/analyze_prompt_signal.py \
      --run_dir output/cifar100/fedta/cifar_hsn/seed_42 \
      --data_path ./local_datasets \
      --out diagnostics_output/prompt_signal_cifar_hsn_s42.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diag_utils import load_args_from_run_dir
from runtime_utils import (clone_prompt_template, load_checkpoint_file,
                           restore_rng_state, rng_state_dict)

EVAL_SEED_BASE = 9700   # house convention (a1_snapshot_eval / hard_example_probe)
RAND_SEED_BASE = 6300   # random-reinit prompt sampling (this script's convention)
NOISE_LEVEL_PP = 1.0    # |delta_acc| < 1pp  =>  "噪声级"


def parse_args():
    p = argparse.ArgumentParser(
        "JE premise probe: per-client prompt ablation on finished runs")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{data_name}/{method}/{run_name}/seed_{seed})")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--out", default=None,
                   help="output JSON (default: diagnostics_output/"
                        "prompt_signal_{run_name}_s{seed}.json)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit_tasks", type=int, default=0,
                   help="analyze only the first N task boundaries (0 = all; debug)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# checkpoint inventory / identity
# ---------------------------------------------------------------------------

def checkpoint_inventory(raw):
    """Structural inventory of a round-end checkpoint (what is stored)."""
    clients = raw.get("clients") or []
    sp = raw.get("server_prompt")
    return {
        "checkpoint_keys": sorted(raw.keys()),
        "has_clients": bool(clients),
        "n_clients": len(clients),
        "client_state_keys": sorted(clients[0].keys()) if clients else None,
        "has_data_split_state": "data_split_state" in raw,
        "has_outer_split_manifest": "outer_split_manifest" in raw,
        "has_server_prompt": sp is not None,
        "server_prompt_keys": sorted(sp.keys()) if sp else None,
        "prompt_pool_shape": list(sp["prompt"].shape) if (sp and "prompt" in sp) else None,
        "prompt_key_shape": list(sp["prompt_key"].shape) if (sp and "prompt_key" in sp) else None,
    }


def state_dicts_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if set(a.keys()) != set(b.keys()):
        return False
    return all(bool(torch.equal(a[k], b[k])) for k in a)


# ---------------------------------------------------------------------------
# bootstrap: models + manifest-driven partition (NO rng replay)
# ---------------------------------------------------------------------------

def build_models(run_args, device):
    """Same model construction as main.build_server (l2p ViT registered)."""
    import Models.vision_transformer__l2p  # noqa: F401  registers vit_base_patch16_224
    from timm.models import create_model

    pretrained_cfg = create_model(run_args.model).default_cfg
    pretrained_cfg['file'] = 'pretrain_model/ViT-B_16.npz'

    original_model = create_model(
        run_args.model, pretrained=True, pretrained_cfg=pretrained_cfg,
        num_classes=run_args.nb_classes, drop_rate=run_args.drop,
        drop_path_rate=run_args.drop_path, drop_block_rate=None)
    model = create_model(
        run_args.model, pretrained=True, pretrained_cfg=pretrained_cfg,
        num_classes=run_args.nb_classes, drop_rate=run_args.drop,
        drop_path_rate=run_args.drop_path, drop_block_rate=None,
        prompt_length=run_args.length, embedding_key=run_args.embedding_key,
        prompt_init=run_args.prompt_key_init, prompt_pool=True,
        prompt_key=run_args.prompt_key, pool_size=run_args.size,
        top_k=run_args.top_k, batchwise_prompt=run_args.batchwise_prompt,
        prompt_key_init=run_args.prompt_key_init, head_type=run_args.head_type,
        use_prompt_mask=run_args.use_prompt_mask)

    original_model.to(device)
    model.to(device)
    if run_args.freeze:
        frozen = tuple(run_args.freeze)
        for n, p in model.named_parameters():
            if n.startswith(frozen):
                p.requires_grad = False
    return original_model, model


def partition_from_manifest(run_args, manifest):
    """Outer federated partition built ONLY from the checkpoint manifest
    (raw sample indices + class masks). No random stream is replayed."""
    from utils import build_transform

    trans = build_transform(True, run_args.input_size)
    if run_args.data_name == 'cifar100':
        from torchvision.datasets import CIFAR100
        from data.cifar100_subset_spliter import CustomedSubset
        trainset = CIFAR100(root=run_args.data_path, train=True, download=False)

        def make_subset(indices):
            return CustomedSubset(trainset, list(indices), trans)
    elif run_args.data_name == 'ImageNet-R':
        from data.continual_datasets import Imagenet_R
        from data.imagenet_r_subset_spliter import CustomedSubset
        trainset = Imagenet_R(root=run_args.data_path, train=True, download=False)

        def make_subset(indices):
            return CustomedSubset(trainset, list(indices), trans, None)
    else:
        raise ValueError(f"unsupported data_name: {run_args.data_name!r}")

    client_data, client_mask = [], []
    for i in range(run_args.client_num):
        per_data, per_mask = [], []
        for t in range(run_args.task_num):
            entry = manifest[str(i)][str(t)]
            per_data.append(make_subset(entry["raw_indices"]))
            per_mask.append(list(entry["class_mask"]))
        client_data.append(per_data)
        client_mask.append(per_mask)
    return client_data, client_mask


def build_server_from_manifest(run_args, raw_ckpt, device):
    from Models.Server_DF import Server_DF

    manifest = raw_ckpt.get("outer_split_manifest")
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(
            "checkpoint missing outer_split_manifest: the federated outer "
            "partition cannot be rebuilt from saved state. Without it none "
            "of the analyses in this script are possible.")
    original_model, model = build_models(run_args, device)
    client_data, client_mask = partition_from_manifest(run_args, manifest)
    server = Server_DF(id='Server', origin_model=original_model,
                       model_name=run_args.model_name,
                       client_num=run_args.client_num,
                       task_num=run_args.task_num, subset=client_data,
                       class_mask=client_mask, lr=run_args.lr,
                       global_epoch=run_args.global_epoch,
                       local_epoch=run_args.local_epoch,
                       batch_size=run_args.batch_size,
                       device=run_args.device, method=run_args.method,
                       threshold=run_args.threshold, surrogate_data=None,
                       test_data=None, args=run_args, model=model,
                       run_manager=None)
    server.init_client()
    server.validate_checkpoint_args(raw_ckpt.get("args", {}))
    if server.build_outer_split_manifest() != manifest:
        raise RuntimeError(
            "manifest-built partition is inconsistent with the checkpoint "
            "manifest (construction bug); refusing to continue.")
    return server


def restore_boundary(server, raw_ckpt):
    """Restore one round-end boundary WITHOUT Server_DF.load_checkpoint.

    Why manual: load_checkpoint rebuilds global_head from clients[0].vit.head
    (a plain Linear) and crashes on skip_fed_avg_head checkpoints, whose
    global_head stays the __init__ Chead (key sets differ). The deployment
    evaluation path never touches global_head, so restoring per-client state
    directly is sufficient and compatible with every Phase-0 checkpoint.
    Inner test loaders are rebuilt from data_split_state (no rng replay)."""
    if raw_ckpt.get("server_model") is not None:
        server.model.load_state_dict(raw_ckpt["server_model"])
    restored = []
    for cs in raw_ckpt.get("clients") or []:
        if cs.get("tail_anchor_model") is None:
            continue
        server.clients[cs["id"]].load_checkpoint_state(cs)
        restored.append(cs["id"])
    server.task_id = raw_ckpt.get("server_task_id", -1)
    return restored


# ---------------------------------------------------------------------------
# evaluation / ablation
# ---------------------------------------------------------------------------

def eval_prompt_mode(client, seen_tasks, nb_classes, mode,
                     server_prompt_state=None):
    """Evaluate `mode` on every seen task of one client.

    The deployment prompt is re-loaded (deepcopy) before every single
    evaluation, so ablations never leak across modes / clients / boundaries.
    The eval seed is fixed per (client, task) so every mode sees identical
    batch order and dropout masks -> delta is purely the ablation effect."""
    per_task = {}
    for t in seen_tasks:
        client.vit.load_prompts(client.prompts)
        pool = client.vit.prompt.prompt
        if mode == "prompt_zero":
            pool.data.zero_()
        elif mode == "prompt_random":
            torch.manual_seed(RAND_SEED_BASE + 100 * client.id + t)
            pool.data.uniform_(-1.0, 1.0)  # Global_Prompt init distribution
        elif mode == "g_prompt_zero":
            template = clone_prompt_template(client.vit)
            template.load_state_dict(server_prompt_state)
            template.prompt.data.zero_()
            client.vit.load_prompts(template)
        # mode "baseline": unmodified deployment prompt
        torch.manual_seed(EVAL_SEED_BASE + 100 * client.id + t)
        per_task[t] = float(client.evaluate(t, nb_classes))
    return per_task


def summarize(per_task, baseline=None):
    accs = list(per_task.values())
    out = {"per_task": {str(t): round(v, 2) for t, v in per_task.items()},
           "mean": round(float(np.mean(accs)), 2)}
    if baseline:
        out["delta_per_task"] = {str(t): round(per_task[t] - baseline[t], 2)
                                 for t in per_task}
        out["delta_mean"] = round(
            float(np.mean(accs)) - float(np.mean(list(baseline.values()))), 2)
    return out


# ---------------------------------------------------------------------------
# prompt key-matching statistics (pure forward, no batchwise voting)
# ---------------------------------------------------------------------------

def l2_normalize(x, dim=1, epsilon=1e-12):
    """Same formula as Global_Prompt.l2_normalize / Tail_Anchor.l2_normalize."""
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(
        torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm


def _usage_stats(counts, n, top1_sims, top2_gaps, pool_size):
    freq = counts.double() / max(int(n), 1)
    nz = freq[freq > 0]
    entropy = float(-(nz * torch.log2(nz)).sum()) if nz.numel() else 0.0
    top_slots = torch.argsort(freq, descending=True)[:10]
    return {
        "n_samples": int(n),
        "distinct_slots": int((counts > 0).sum()),
        "usage_entropy_bits": round(entropy, 4),
        "usage_entropy_norm": (round(entropy / math.log2(pool_size), 4)
                               if pool_size > 1 else None),
        "top1_slot_share": round(float(freq.max()), 4),
        "top10_slots": [{"slot": int(s), "freq": round(float(freq[s]), 4)}
                        for s in top_slots],
        "mean_top1_similarity": (round(float(torch.cat(top1_sims).mean()), 4)
                                 if top1_sims else None),
        "mean_top1_top2_gap": (round(float(torch.cat(top2_gaps).mean()), 4)
                               if top2_gaps else None),
    }


def prompt_key_stats(client, seen_tasks, device):
    """Per-sample top-k prompt-slot selection stats on the client's inner
    test sets, under the deployment prompt's key rows. This is the raw
    key-matching signal (batchwise_prompt majority voting NOT applied)."""
    client.vit.load_prompts(client.prompts)
    prompt = client.vit.prompt
    pool_size = int(prompt.pool_size)
    top_k = int(prompt.top_k)
    key = prompt.prompt_key.detach()
    if key.dim() > 2:
        key = key.reshape(-1, key.shape[-1])
    key_norm = l2_normalize(key, dim=1).to(device)

    per_task, total_counts = {}, None
    top1_sims, top2_gaps, total_n = [], [], 0
    for t in seen_tasks:
        loader = DataLoader(client.test_loader[t], batch_size=16,
                            shuffle=False, num_workers=0)
        counts = torch.zeros(pool_size, dtype=torch.long)
        sims, gaps, n = [], [], 0
        for x, _target in loader:
            x = x.to(device)
            with torch.no_grad():
                cls = client.original_model(x)["pre_logits"]
            sim = l2_normalize(cls, dim=1) @ key_norm.t()  # B, pool_size
            k = min(max(top_k, 2), pool_size)
            vals, idx = torch.topk(sim, k=k, dim=1)
            sel = idx[:, :top_k].reshape(-1).cpu()
            counts += torch.bincount(sel, minlength=pool_size)
            sims.append(vals[:, 0].cpu())
            if vals.shape[1] >= 2:
                gaps.append((vals[:, 0] - vals[:, 1]).cpu())
            n += int(x.shape[0])
        per_task[str(t)] = _usage_stats(counts, n, sims, gaps, pool_size)
        total_counts = counts if total_counts is None else total_counts + counts
        top1_sims.extend(sims)
        top2_gaps.extend(gaps)
        total_n += n
    return {"pool_size": pool_size, "top_k": top_k,
            "note": "per-sample top-k key matching (l2-normalized cosine, "
                    "no batchwise voting)",
            "aggregate": _usage_stats(total_counts, total_n, top1_sims,
                                      top2_gaps, pool_size),
            "per_task": per_task}


# ---------------------------------------------------------------------------
# per-boundary analysis
# ---------------------------------------------------------------------------

def analyze_client(client, raw_state, run_args, device, server_prompt_state):
    nb = run_args.nb_classes
    seen = [t for t in range(client.task_id + 1)
            if t < len(client.heads) and client.heads[t] is not None
            and t < len(client.test_loader)]
    out = {"task_id": client.task_id, "seen_tasks": seen,
           "n_test_samples": {str(t): len(client.test_loader[t]) for t in seen}}
    if not seen:
        out["limitation"] = ("no evaluable seen task (heads / data_split_state "
                             "missing for this client)")
        return out
    if client.prompts is None:
        out["limitation"] = ("no prompt state in checkpoint: baseline "
                             "semantics undefined, prompt ablation and key "
                             "stats skipped")
        return out

    base = eval_prompt_mode(client, seen, nb, "baseline")
    zero = eval_prompt_mode(client, seen, nb, "prompt_zero")
    rand = eval_prompt_mode(client, seen, nb, "prompt_random")

    out["baseline"] = summarize(base)
    out["local_prompt_zero"] = summarize(zero, base)
    out["local_prompt_random"] = summarize(rand, base)

    raw_prompts = (raw_state or {}).get("prompts")
    if server_prompt_state is None:
        out["g_prompt_zero"] = {"limitation": "server_prompt missing in checkpoint"}
    elif raw_prompts is not None and state_dicts_equal(raw_prompts, server_prompt_state):
        # broadcast already synced this client to the fused server prompt:
        # zeroing it is the identical operation -> reuse, flagged
        out["g_prompt_zero"] = {"same_weights_as_local_zero": True,
                                **summarize(zero, base)}
    else:
        g = eval_prompt_mode(client, seen, nb, "g_prompt_zero",
                             server_prompt_state)
        out["g_prompt_zero"] = {"same_weights_as_local_zero": False,
                                **summarize(g, base)}

    out["key_stats"] = prompt_key_stats(client, seen, device)
    return out


def analyze_boundary(server, raw_ckpt, run_args, device):
    restored = restore_boundary(server, raw_ckpt)
    sp = raw_ckpt.get("server_prompt")
    cstates = raw_ckpt.get("clients") or []

    b = {"checkpoint": str(raw_ckpt.get("completed_round")),
         "completed_round": raw_ckpt.get("completed_round"),
         "after_task": server.task_id,
         "restored_clients": restored}

    # prompt identity audit (raw checkpoint states, byte-exact)
    prompts = [cs.get("prompts") for cs in cstates]
    with_prompt = [p for p in prompts if p is not None]
    sm = raw_ckpt.get("server_model") or {}
    sm_prompt = {k[len("prompt."):]: v for k, v in sm.items()
                 if k.startswith("prompt.")}
    b["prompt_identity"] = {
        "n_client_prompt_states": len(with_prompt),
        "all_clients_identical": bool(
            with_prompt and len(with_prompt) == len(prompts)
            and all(state_dicts_equal(p, with_prompt[0]) for p in with_prompt)),
        "clients_match_server_prompt": bool(
            sp is not None and with_prompt and len(with_prompt) == len(prompts)
            and all(state_dicts_equal(p, sp) for p in with_prompt)),
        "server_model_prompt_matches_server_prompt": bool(
            sm_prompt and sp is not None
            and state_dicts_equal(sm_prompt, sp)),
    }
    if not cstates:
        b["limitation"] = ("checkpoint has no client states: no per-client "
                           "analysis possible (G_prompt ablation and key "
                           "stats need per-client heads/test sets too)")

    clients_out = {}
    for cs in cstates:
        cid = cs["id"]
        if cid not in restored:
            clients_out[str(cid)] = {
                "limitation": "tail_anchor_model missing in checkpoint; "
                              "client skipped"}
            continue
        clients_out[str(cid)] = analyze_client(
            server.clients[cid], cs, run_args, device, sp)
    b["clients"] = clients_out

    def _noise(field):
        ds = [c[field]["delta_mean"] for c in clients_out.values()
              if isinstance(c, dict) and field in c
              and isinstance(c[field], dict) and "delta_mean" in c[field]]
        return {"n_clients": len(ds),
                "n_noise_level": sum(1 for d in ds if abs(d) < NOISE_LEVEL_PP),
                "mean_delta": round(float(np.mean(ds)), 2) if ds else None}

    b["summary"] = {"local_prompt_zero": _noise("local_prompt_zero"),
                    "local_prompt_random": _noise("local_prompt_random")}
    return b


def print_boundary_table(name, b):
    ident = b.get("prompt_identity", {})
    print(f"\n----- {name} (after_task={b.get('after_task')}) | "
          f"prompt identity: clients==server_prompt="
          f"{ident.get('clients_match_server_prompt')} -----")
    print(f"{'client':>6} {'base':>7} {'zero':>7} {'d_zero':>7} "
          f"{'rand':>7} {'d_rand':>7} {'keyH(b)':>8} {'Hnorm':>6} "
          f"{'slots':>5} {'top1%':>6} {'n':>6}")
    for cid in sorted(b["clients"], key=int):
        c = b["clients"][cid]
        if not isinstance(c, dict) or "baseline" not in c:
            lim = c.get("limitation", "skipped") if isinstance(c, dict) else "skipped"
            print(f"{cid:>6}  -- {lim}")
            continue
        ks = c["key_stats"]["aggregate"]
        hnorm = ks.get("usage_entropy_norm")
        hnorm = hnorm if hnorm is not None else float("nan")
        print(f"{cid:>6} "
              f"{c['baseline']['mean']:>7.2f} "
              f"{c['local_prompt_zero']['mean']:>7.2f} "
              f"{c['local_prompt_zero']['delta_mean']:>+7.2f} "
              f"{c['local_prompt_random']['mean']:>7.2f} "
              f"{c['local_prompt_random']['delta_mean']:>+7.2f} "
              f"{ks['usage_entropy_bits']:>8.3f} "
              f"{hnorm:>6.3f} "
              f"{ks['distinct_slots']:>5d} "
              f"{100 * ks['top1_slot_share']:>6.1f} "
              f"{ks['n_samples']:>6d}")
    s = b["summary"]
    print(f"noise-level (|delta|<{NOISE_LEVEL_PP}pp): "
          f"zero {s['local_prompt_zero']['n_noise_level']}/{s['local_prompt_zero']['n_clients']}, "
          f"random {s['local_prompt_random']['n_noise_level']}/"
          f"{s['local_prompt_random']['n_clients']}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    device = args.device

    run_args = load_args_from_run_dir(run_dir, data_path=args.data_path,
                                      device=device)
    if run_args.data_name == 'cifar100':
        run_args.nb_classes = 100
    elif run_args.data_name == 'ImageNet-R':
        run_args.nb_classes = 200
    else:
        raise ValueError(f"unsupported data_name: {run_args.data_name!r}")

    ckpt_dir = run_dir / "checkpoints"
    boundaries = sorted(ckpt_dir.glob("task_*_end.pth"))
    if args.limit_tasks > 0:
        boundaries = boundaries[:args.limit_tasks]
    if not boundaries:
        latest = ckpt_dir / "latest.pth"
        if not latest.is_file():
            raise FileNotFoundError(
                f"no task_*_end.pth or latest.pth under {ckpt_dir}")
        boundaries = [latest]

    out_path = (Path(args.out) if args.out else
                Path("diagnostics_output") /
                f"prompt_signal_{run_dir.parent.name}_s{run_args.seed}.json")

    first_raw = load_checkpoint_file(str(boundaries[0]))
    inv = checkpoint_inventory(first_raw)
    print("[inventory] checkpoint keys:", ", ".join(inv["checkpoint_keys"]))
    if inv["client_state_keys"]:
        print("[inventory] client state keys:",
              ", ".join(inv["client_state_keys"]))
    print(f"[inventory] server_prompt keys: {inv['server_prompt_keys']} "
          f"pool{inv['prompt_pool_shape']} key{inv['prompt_key_shape']}")
    print(f"[inventory] data_split_state={inv['has_data_split_state']} "
          f"outer_split_manifest={inv['has_outer_split_manifest']} "
          f"n_clients={inv['n_clients']}")

    server = build_server_from_manifest(run_args, first_raw, device)
    del first_raw

    results = {
        "meta": {
            "script": "diagnostics/analyze_prompt_signal.py",
            "run_dir": str(run_dir),
            "data_name": run_args.data_name,
            "device": device,
            "client_num": run_args.client_num,
            "task_num": run_args.task_num,
            "nb_classes": run_args.nb_classes,
            "eval_semantics": (
                "Client_DF.evaluate verbatim: inner 30% test split, batch=8 "
                "shuffle=True, original_model -> vit(prompt) -> Tail_Anchor "
                "-> task Chead, class_mask -inf masking; identical seed per "
                "(client, task) across modes so delta is purely the ablation "
                "(official evaluate keeps train-mode Chead dropout, seeded)"),
            "seed_convention": {"eval": "9700+100*client+task",
                                "random_ablation": "6300+100*client+task"},
            "ablation": {
                "prompt_zero": "prompt pool -> 0 (prompt_key kept, routing "
                               "unchanged)",
                "prompt_random": "prompt pool ~ U(-1,1) (Global_Prompt init "
                                 "distribution), key kept",
                "g_prompt_zero": "server_prompt (SIKF fused) pool -> 0"},
            "noise_level_pp": NOISE_LEVEL_PP,
            "checkpoint_inventory": inv,
            "boundaries": [p.name for p in boundaries],
        },
        "boundaries": {},
    }

    rng_snapshot = rng_state_dict()
    try:
        for path in boundaries:
            print(f"\n===== boundary checkpoint: {path.name} =====")
            raw = load_checkpoint_file(str(path))
            try:
                b = analyze_boundary(server, raw, run_args, device)
            finally:
                del raw
            label = f"after_task_{b['after_task']}"
            results["boundaries"][label] = b
            print_boundary_table(path.name, b)
    finally:
        restore_rng_state(rng_snapshot)

    # conclusion on the FINAL analyzed boundary
    if not results["boundaries"]:
        conclusion = "no boundary could be analyzed (see per-boundary errors)"
    else:
        final_key = list(results["boundaries"])[-1]
        fb = results["boundaries"][final_key]
        sz = fb["summary"]["local_prompt_zero"]
        sr = fb["summary"]["local_prompt_random"]
        if sz["n_clients"] == 0:
            conclusion = (f"{final_key}: no evaluable client "
                          f"(see limitations in JSON)")
        else:
            noise = (sz["n_noise_level"] * 2 > sz["n_clients"])
            verdict = ("多数客户端 prompt 置零 |Δacc|<1pp -> prompt 组件贡献为"
                       "噪声级，JE 前提（该维度）在该 run 上成立" if noise else
                       "噪声级客户端未过半 -> JE 前提（该维度）在该 run 上"
                       "不成立/需复核")
            conclusion = (
                f"{final_key}: prompt 置零噪声级客户端 "
                f"{sz['n_noise_level']}/{sz['n_clients']}"
                f"（mean Δ={sz['mean_delta']}pp）；随机替换 "
                f"{sr['n_noise_level']}/{sr['n_clients']}。{verdict}。")
    results["conclusion"] = conclusion
    print("\n[结论] " + conclusion)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
