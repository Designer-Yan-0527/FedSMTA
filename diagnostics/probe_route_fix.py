"""A2-R route-revival probe: soft routing (+key reset) micro fine-tune on a
finished FedTA run checkpoint.

背景（诊断结论，见 analyze_prompt_signal.py 的 key_stats 证据）:
官方 FedTA（CVPR2025）的 Global_Prompt 路由结构性失效 —— prompt_key 池的
keyH ≈ 0.004 bits，所有样本 100% 命中单一槽，prompt 退化为固定 bias。死锁
三环节:
  (1) top-k/argmax 选择不可导，路由决定本身无梯度；
  (2) uniform(-1,1) 随机初始化 + pull 只拉选中槽 → 初始垄断正反馈固化；
  (3) 单槽命中下训练目标退化为均值。

拟议修复 A2-R = R1(key 球面正交重置) + R2(可导软路由 softmax(sim/τ)) +
R3(槽位熵正则)。本探针只动 R1+R2 两个变量，**不含 R3**（一次探针只动两个
变量）。

探针语义（模拟"部署后修复"，不进主线训练协议）:
  从已完成 run 的 round-end checkpoint 出发，对每个 client 独立地
    1. 按 analyze_prompt_signal 的组装方式恢复该 client 完整模型
       (共享 ViT+prompt 池 + 该 client 的 Tail_Anchor/heads/prompts/
       test_loader，全部走 checkpoint 保存状态，不重放随机流)；
    2. 微调前基线: 该 client 全部 seen 任务 inner test 准确率(官方
       Client_DF.evaluate 语义) + key_stats(复用观察者算法)；
    3. 运行时把 Global_Prompt.forward 的 top-k 硬选择替换为可导软路由
       w = softmax(sim/τ) 对池槽加权混合（训练与推理同款软加权；
       prompt 拼接位置/长度不变）。**不改 Models/ 下任何文件**；
    4. --arm reset (R1+R2): prompt_key 行正交化重置(球面均匀方向);
       --arm keep  (R2 only): 保留 checkpoint 的 key 不重置;
    5. 微调: 解冻 prompt 池 + prompt_key + 该 client 全部 seen 任务的
       Chead heads（Tail_Anchor 的 key/anchor_pool 与 ViT 骨架一律冻结，
       冻结方式 = optimizer 只收 trainable 集合）。训练数据 = 该 client
       全部 seen 任务 inner train 混合(shuffle)。Adam(lr, wd=1e-3)，
       损失 = 官方同款 CE(masked logits, target) − 0.1×pull_off，
       pull_off 用软路由下的 reduce_sim（语义兼容，见下）；
    6. 微调后同款测量（after 在软路由激活下测 —— 探针部署形态即软路由）。

reduce_sim 兼容性结论（docstring 要求注明）:
  官方 reduce_sim = mean_b(Σ_{k∈top-k} cos(x_b, key_sel))；软路由版本 =
  top_k · mean_b(Σ_s w[b,s]·cos(x_b, key_s))。两者同为"选中 key 与 query
  的平均余弦、越大越惩罚 CE − 0.1×pull 中的 −0.1 项"，量级经 top_k 对齐，
  且对 prompt_key 经 w 可导 —— 语义兼容，因此**沿用 CE − 0.1×pull，不需要
  退化为纯 CE**。

patch 覆盖面说明（必读文件 4 的验证结论）:
  vision_transformer__l2p.VisionTransformer.forward_features 中
  `self.prompt(x, prompt_mask=..., cls_features=...)` 是 Global_Prompt.forward
  的唯一调用点，训练 (Client_DF.train) 与推理 (Client_DF.evaluate) 都走
  vit(...) → forward_features → self.prompt(...)，因此**类级替换
  Global_Prompt.forward 即覆盖全部路径**，无需 patch 调用方。
  本 run use_prompt_mask=False、batchwise_prompt=True：软路由按样本加权，
  天然取代 batchwise 批内投票分支；若 prompt_mask 非 None 则 fail-closed
  （本 run 不会发生）。key_stats 观察者算法独立复算相似度，不受 patch 影响。

两臂隔离（内存消融安全）:
  --arm both 时顺序执行 keep → reset；**每臂开始前都从常驻内存的 raw
  checkpoint 重新 restore_boundary**（server.model + 全部 client 的
  Tail_Anchor/heads/prompts/test_loader 全量重载，checkpoint 文件只读），
  微调只发生在 per-client deepcopy 出的 prompt 副本与 client 私有 head
  对象上，共享 vit/共享 original_model 不被持久污染；两臂的 before 基线
  各自重测且应逐字节一致（isolation_check 校验并写进 JSON）。

判读标准:
  路由复活: after keyH − before keyH > 0.5 bits（>0.5 bits 算复活）；
  准确率守住: 客户端平均 Δacc(mean) ≥ −2pp 且无任何单任务掉幅 > 10pp。

已知边界（如实记录）:
  - 该 run 的 latest.pth 恢复后 task_id=2（进行中的 run，seen=3 任务，
    脚本按 checkpoint 实际恢复状态自适应，不硬编码 5 任务）；
  - Tail_Anchor.forward 内部硬编码 `.to('cuda')`，--device 非 cuda 会崩
    （继承自主线代码的限制）；
  - Chead 输出 200 维但官方按 nb_classes=100 掩码，finetune 掩码公式与
    官方逐字一致（setdiff1d(arange(nb_classes), class_mask[task])，
    混合批次下按样本各自 task 的 class_mask 应用）。

Usage (repo root):
  python diagnostics/probe_route_fix.py \
      --run_dir output/cifar100/fedta_a2/cifar_a2_key_hsn/seed_42 \
      --data_path ./local_datasets --device cuda --arm both --tau 0.1
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from diag_utils import load_args_from_run_dir, resolve_checkpoint
from runtime_utils import load_checkpoint_file
from analyze_prompt_signal import (EVAL_SEED_BASE, checkpoint_inventory,
                                   build_server_from_manifest,
                                   restore_boundary, prompt_key_stats,
                                   l2_normalize)
from Models.global_prompt import Global_Prompt

WEIGHT_DECAY = 1e-3      # official Client_DF optimizer: Adam(lr, wd=1e-3)
PULL_COEFF = 0.1         # official phase-1 loss: CE - 0.1 * pull_off
FT_SEED_BASE = 6600      # per-(arm, client) fine-tune RNG stream (this script)
KEY_RESET_SEED_BASE = 6700  # R1 key reset independent generator stream
KEYH_REVIVAL_BITS = 0.5  # route revival threshold (bits)
MEAN_DROP_TOL_PP = 2.0   # mean-acc drop tolerance (percentage points)
TASK_DROP_TOL_PP = 10.0  # worst single-task drop tolerance (pp)


def parse_args():
    p = argparse.ArgumentParser(
        "A2-R route-revival probe: soft routing (+ key reset) micro "
        "fine-tune on a finished run")
    p.add_argument("--run_dir",
                   default="output/cifar100/fedta_a2/cifar_a2_key_hsn/seed_42",
                   help="run directory (.../{data_name}/{method}/{run_name}/seed_{seed})")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3,
                   help="probe fine-tune epochs per client")
    p.add_argument("--batch_size", type=int, default=16,
                   help="probe fine-tune batch size (official local batch)")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--tau", type=float, default=0.1,
                   help="soft routing temperature: w = softmax(sim/tau)")
    p.add_argument("--arm", choices=["reset", "keep", "both"], default="both",
                   help="reset = R1(key reset)+R2(soft route); keep = R2 only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers (default: run_args.num_workers)")
    p.add_argument("--checkpoint", default=None,
                   help="boundary checkpoint (default: run_dir/checkpoints/latest.pth; "
                        "also accepts a task_XX_end.pth)")
    p.add_argument("--out",
                   default="diagnostics_output/route_fix_probe_cifar_a2key_hsn_s42.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# R2: soft-routing patch (runtime class-level swap; Models/ files untouched)
# ---------------------------------------------------------------------------

_SOFT_STATE = {"tau": None}
_ORIGINAL_PROMPT_FORWARD = Global_Prompt.forward


def _soft_route_forward(self, x_embed, prompt_mask=None, cls_features=None):
    """R2 drop-in replacement for Global_Prompt.forward.

    Everything before slot selection is copied verbatim from the official
    forward (embedding_key branches / l2-normalize / cosine similarity).
    The hard top-k + batchwise majority vote (+ prompt_mask) selection block
    is replaced by a differentiable softmax over ALL pool slots:

        w              = softmax(similarity / tau)          (B, pool)
        batched_prompt = einsum('bp,plc->blc', w, prompt)    (B, length, C)

    Token count is preserved exactly (total_prompt_len == length * top_k):
    when top_k > 1 the mixed prompt is repeated top_k times (top_k == 1 in
    this run -> identity). Prompt concatenation position and all other
    outputs stay unchanged, so pos_embed / head_type='prompt' consumers are
    unaffected.

    reduce_sim soft analogue (pull term):
        official = mean_b( sum_{k in top-k} cos(x_b, key_sel) )
        soft     = top_k * mean_b( sum_s w[b,s] * cos(x_b, key_s) )
    scale-matched (soft weights sum to 1, times top_k == sum of top-k picks)
    and differentiable in prompt_key through w -> CE - 0.1*pull stays valid.
    """
    tau = _SOFT_STATE["tau"]
    if tau is None:
        raise RuntimeError("soft route installed but tau unset (internal bug)")
    out = dict()
    if self.prompt_pool:
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError(
                "Not supported way of calculating embedding keys!")

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1)    # Pool, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)      # B, C
        similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool

        if prompt_mask is not None:
            raise RuntimeError(
                "probe soft routing is incompatible with use_prompt_mask; "
                "this run must have use_prompt_mask=False")

        w = torch.softmax(similarity / tau, dim=-1)               # B, Pool
        batched_prompt = torch.einsum('bp,plc->blc', w, self.prompt)  # B,L,C
        if self.top_k > 1:
            batched_prompt = batched_prompt.repeat(1, self.top_k, 1)

        out['prompt_idx'] = torch.argmax(similarity, dim=1)  # diagnostic only
        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity
        out['soft_weights'] = w
        reduce_sim = (self.top_k * torch.sum(w * similarity)
                      / x_embed.shape[0])
        out['reduce_sim'] = reduce_sim
    else:
        # non-pool branch copied verbatim from the official forward
        if self.prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
        elif self.prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
            nn.init.uniform_(self.prompt)
        batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

    out['total_prompt_len'] = batched_prompt.shape[1]
    out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
    return out


def install_soft_route(tau):
    _SOFT_STATE["tau"] = float(tau)
    if Global_Prompt.forward is not _soft_route_forward:
        Global_Prompt.forward = _soft_route_forward


def uninstall_soft_route():
    _SOFT_STATE["tau"] = None
    Global_Prompt.forward = _ORIGINAL_PROMPT_FORWARD


# ---------------------------------------------------------------------------
# R1: orthonormal prompt_key reset (spherical-uniform + row-orthogonalized)
# ---------------------------------------------------------------------------

def reset_prompt_key_orthonormal(prompt_module, seed):
    """R1: replace prompt_key rows by mutually-orthogonal unit directions.

    Implementation: QR of a Gaussian matrix == nn.init.orthogonal_ (Haar
    measure on the sphere), drawn from an INDEPENDENT torch.Generator for
    reproducibility; rows then L2-normalized with the same formula the
    forward uses. Since forward l2-normalizes both key and cls features,
    the key's absolute scale is irrelevant -- only directions matter.
    """
    key = prompt_module.prompt_key
    if not isinstance(key, torch.nn.Parameter):
        raise RuntimeError(
            "R1 key reset requires a learnable prompt_key (prompt_key=True)")
    pool_size, dim = int(key.shape[0]), int(key.shape[1])
    if pool_size > dim:
        raise RuntimeError(
            f"orthonormal rows need pool_size<=embed_dim, got {pool_size}>{dim}")
    gen = torch.Generator()  # independent stream, CPU-side
    gen.manual_seed(int(seed))
    raw = torch.randn(dim, pool_size, generator=gen)
    q, _ = torch.linalg.qr(raw)          # (dim, pool), orthonormal columns
    rows = q.t().contiguous()           # (pool, dim), orthonormal rows
    rows = l2_normalize(rows, 1)         # safeguard; same space as forward
    key.data.copy_(rows.to(key.device))


# ---------------------------------------------------------------------------
# measurement (official evaluate semantics + observer key_stats algorithm)
# ---------------------------------------------------------------------------

def compute_seen(client):
    """Observer's rule: tasks with restored heads AND test loaders."""
    return [t for t in range(client.task_id + 1)
            if t < len(client.heads) and client.heads[t] is not None
            and t < len(client.test_loader)]


def measure_client(client, seen, nb_classes, device):
    """Per-task inner-test accuracy + key_stats on the CURRENT deployed state.

    Eval seeds are fixed per (client, task) (house convention) so before and
    after see identical batch order / dropout masks -> delta is attributable
    to the arm itself. Must be called with the intended routing state
    (before: patch OFF; after: patch ON -- the probe's deployed form)."""
    accs = {}
    for t in seen:
        client.vit.load_prompts(client.prompts)
        torch.manual_seed(EVAL_SEED_BASE + 100 * client.id + t)
        accs[str(t)] = round(float(client.evaluate(t, nb_classes)), 2)
    mean_acc = round(float(np.mean(list(accs.values()))), 2)
    key_stats = prompt_key_stats(client, seen, device)
    return {"acc": accs, "mean_acc": mean_acc, "key_stats": key_stats}


# ---------------------------------------------------------------------------
# probe fine-tune
# ---------------------------------------------------------------------------

class _TaskTaggedDataset(Dataset):
    """(x, y, task_id) wrapper over a checkpoint-recorded inner-train split."""

    def __init__(self, base_dataset, indices, task_id):
        self._subset = Subset(base_dataset, list(indices))
        self.task_id = task_id

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, i):
        x, y = self._subset[i]
        return x, y, self.task_id


def finetune_client(client, seen, probe, run_args, device, arm, arm_idx):
    """Micro fine-tune one client under soft routing.

    Trainable = vit.prompt (pool + key) ∪ {heads[t] : t in seen}. Everything
    else is frozen by optimizer scope (backbone was already requires_grad=False
    via run_args.freeze at build time; Tail_Anchor key/anchor_pool and vit.head
    receive no update). Heads are assigned into Tail_Anchor WITHOUT deepcopy
    so gradients land in the deployed heads[t] objects themselves.
    Loss = CE(masked logits, target) - 0.1 * soft reduce_sim (official coeff).
    """
    client.original_model.eval()           # official train() convention
    client.vit.load_prompts(client.prompts)  # fresh deepcopy of deployed prompt
    client.vit.to(device)
    client.model.to(device)
    for t in seen:
        client.heads[t].to(device)

    if arm == "reset":
        reset_seed = (KEY_RESET_SEED_BASE + 1000 * (probe.seed % 1000)
                      + 100 * arm_idx + client.id)
        reset_prompt_key_orthonormal(client.vit.prompt, reset_seed)
        print(f"[R1] client {client.id}: prompt_key rows re-orthonormalized "
              f"(seed={reset_seed})")

    nb = run_args.nb_classes
    prompt_module = client.vit.prompt
    trainable = list(prompt_module.parameters())
    for t in seen:
        trainable += list(client.heads[t].parameters())
    optimizer = torch.optim.Adam(trainable, lr=probe.lr,
                                 weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parts = [_TaskTaggedDataset(client.train_data[t],
                                 client.data_split_indices[t]["train"], t)
             for t in seen]
    n_train = {str(t): len(parts[i]) for i, t in enumerate(seen)}
    loader = DataLoader(ConcatDataset(parts), batch_size=probe.batch_size,
                         shuffle=True, num_workers=probe.num_workers)

    # official masking formula, cached per task (applied per sample in mixed batches)
    row_mask = {}
    orig_head = client.model.head
    try:
        for epoch in range(probe.epochs):
            ce_sum, pull_sum, n_batches = 0.0, 0.0, 0
            for x, y, tid in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long()
                tid = tid.to(device).long()
                with torch.no_grad():
                    cls_features = client.original_model(x)["pre_logits"]
                out = client.vit(x, task_id=client.task_id,
                                 cls_features=cls_features, train=True)
                feat = out["feat"]
                pull = out["reduce_sim"]

                logits = None
                allow = None
                for t in torch.unique(tid).tolist():
                    sel = (tid == t)
                    client.model.head = client.heads[t]  # no deepcopy: grads persist
                    pre, _x1, _ta_pull = client.model(feat[sel], None)
                    if logits is None:
                        logits = torch.zeros(len(y), pre.shape[1], device=device)
                        allow = torch.zeros_like(logits, dtype=torch.bool)
                    logits[sel] = pre
                    if t not in row_mask:
                        r = torch.ones(pre.shape[1], dtype=torch.bool,
                                        device=device)
                        not_mask = np.setdiff1d(np.arange(nb),
                                                list(client.class_mask[t]))
                        r[torch.tensor(not_mask, dtype=torch.long,
                                       device=device)] = False
                        row_mask[t] = r
                    allow[sel] = row_mask[t]

                logits = logits.masked_fill(~allow, float("-inf"))
                ce = criterion(logits, y)
                loss = ce - PULL_COEFF * pull
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ce_sum += float(ce.detach())
                pull_sum += float(pull.detach())
                n_batches += 1
            print(f"[finetune] arm={arm} client={client.id} epoch={epoch}: "
                  f"mean_CE={ce_sum / max(n_batches, 1):.4f} "
                  f"mean_pull={pull_sum / max(n_batches, 1):.4f}")
    finally:
        client.model.head = orig_head  # hygiene: leave Tail_Anchor as restored

    # persist the trained prompt into the client's deployed state
    client.prompts = deepcopy(client.vit.get_prompts())
    return {"n_train": n_train}


# ---------------------------------------------------------------------------
# per-arm driver
# ---------------------------------------------------------------------------

def run_arm(arm, arm_idx, server, raw, run_args, probe, device):
    """One arm = full checkpoint re-restore -> per-client probe -> record.

    The re-restore at arm start is the isolation mechanism: every mutation of
    the previous arm lived on per-client copies / client-local objects, and
    this reload rewrites all of them from the raw checkpoint state."""
    restore_boundary(server, raw)
    nb = run_args.nb_classes
    clients_out = {}
    for cs in raw.get("clients") or []:
        cid = cs["id"]
        if cs.get("tail_anchor_model") is None:
            clients_out[str(cid)] = {
                "limitation": "tail_anchor_model missing in checkpoint; skipped"}
            continue
        client = server.clients[cid]
        seen = compute_seen(client)
        if client.prompts is None or not seen:
            clients_out[str(cid)] = {
                "limitation": "no evaluable seen task / no prompt state; skipped"}
            continue

        before = measure_client(client, seen, nb, device)  # hard routing
        install_soft_route(probe.tau)
        try:
            ft_info = finetune_client(client, seen, probe, run_args, device,
                                      arm, arm_idx)
            after = measure_client(client, seen, nb, device)  # soft routing
        finally:
            uninstall_soft_route()

        clients_out[str(cid)] = {
            "task_id": client.task_id,
            "seen_tasks": seen,
            "n_train": ft_info["n_train"],
            "before": before,
            "after": after,
        }
    return clients_out


# ---------------------------------------------------------------------------
# verdict / reporting
# ---------------------------------------------------------------------------

def arm_summary(clients_out):
    per = []
    for cid, rec in clients_out.items():
        if not isinstance(rec, dict) or "before" not in rec:
            continue
        b, a = rec["before"], rec["after"]
        d_keyh = (a["key_stats"]["aggregate"]["usage_entropy_bits"]
                  - b["key_stats"]["aggregate"]["usage_entropy_bits"])
        d_acc = a["mean_acc"] - b["mean_acc"]
        worst = min(a["acc"][t] - b["acc"][t] for t in b["acc"])
        per.append({"client": cid, "d_keyH": round(d_keyh, 4),
                    "d_acc": round(d_acc, 2), "worst_drop": round(worst, 2)})
    if not per:
        return {"n_clients": 0}
    mean_d = float(np.mean([p["d_acc"] for p in per]))
    worst_all = min(p["worst_drop"] for p in per)
    return {
        "n_clients": len(per),
        "n_keyH_revived": sum(1 for p in per if p["d_keyH"] > KEYH_REVIVAL_BITS),
        "mean_delta_acc": round(mean_d, 2),
        "worst_task_drop": round(worst_all, 2),
        "acc_held": bool(mean_d >= -MEAN_DROP_TOL_PP
                         and worst_all >= -TASK_DROP_TOL_PP),
        "per_client": per,
    }


def print_arm_table(arm, clients_out):
    title = ("R1(key reset) + R2(soft route)" if arm == "reset"
             else "R2(soft route) only")
    print(f"\n===== arm={arm} ({title}) =====")
    print(f"{'client':>6} {'keyH_b':>8} {'keyH_a':>8} {'dH':>7} "
          f"{'slots_b':>7} {'slots_a':>7} {'top1_b%':>8} {'top1_a%':>8} "
          f"{'acc_b':>7} {'acc_a':>7} {'d_acc':>7} {'worst':>7}")
    for cid in sorted(clients_out, key=int):
        rec = clients_out[cid]
        if not isinstance(rec, dict) or "before" not in rec:
            print(f"{cid:>6}  -- {rec.get('limitation', 'skipped')}")
            continue
        b, a = rec["before"], rec["after"]
        kb = b["key_stats"]["aggregate"]
        ka = a["key_stats"]["aggregate"]
        dH = ka["usage_entropy_bits"] - kb["usage_entropy_bits"]
        worst = min(a["acc"][t] - b["acc"][t] for t in b["acc"])
        print(f"{cid:>6} "
              f"{kb['usage_entropy_bits']:>8.3f} {ka['usage_entropy_bits']:>8.3f} "
              f"{dH:>+7.3f} "
              f"{kb['distinct_slots']:>7d} {ka['distinct_slots']:>7d} "
              f"{100 * kb['top1_slot_share']:>8.1f} {100 * ka['top1_slot_share']:>8.1f} "
              f"{b['mean_acc']:>7.2f} {a['mean_acc']:>7.2f} "
              f"{a['mean_acc'] - b['mean_acc']:>+7.2f} {worst:>+7.2f}")
    s = arm_summary(clients_out)
    if s["n_clients"]:
        print(f"arm={arm}: keyH 复活 {s['n_keyH_revived']}/{s['n_clients']} "
              f"(Δ>0.5bits), 平均Δacc {s['mean_delta_acc']:+.2f}pp, "
              f"最差单任务 {s['worst_task_drop']:+.2f}pp, "
              f"准确率{'守住' if s['acc_held'] else '受损'}")


def isolation_diff(arms_out):
    """Both arms' before baselines must be identical (same checkpoint state +
    fixed eval seeds). Any difference = an isolation bug, reported loudly."""
    if not ("keep" in arms_out and "reset" in arms_out):
        return None
    common = [cid for cid in arms_out["keep"]
              if cid in arms_out["reset"]
              and "before" in arms_out["keep"][cid]
              and "before" in arms_out["reset"][cid]]
    worst = 0.0
    for cid in common:
        bk = arms_out["keep"][cid]["before"]
        br = arms_out["reset"][cid]["before"]
        for t in bk["acc"]:
            worst = max(worst, abs(bk["acc"][t] - br["acc"][t]))
        worst = max(worst, abs(bk["key_stats"]["aggregate"]["usage_entropy_bits"]
                               - br["key_stats"]["aggregate"]["usage_entropy_bits"]))
    return {"n_common_clients": len(common),
            "max_abs_diff": round(worst, 6),
            "before_identical_across_arms": bool(worst == 0.0)}


def build_conclusion(summaries):
    parts = []
    for arm in ("keep", "reset"):
        s = summaries.get(arm)
        if not s or not s.get("n_clients"):
            parts.append(f"{arm}: 无可评估客户端")
            continue
        parts.append(
            f"{arm}: 软路由使 keyH 显著上升(>0.5bits) {s['n_keyH_revived']}/{s['n_clients']}, "
            f"平均Δacc {s['mean_delta_acc']:+.2f}pp, 最差单任务 "
            f"{s['worst_task_drop']:+.2f}pp, "
            f"准确率{'守住' if s['acc_held'] else '未守住'}")
    r = summaries.get("reset") or {}
    if r.get("n_clients"):
        if r["n_keyH_revived"] == r["n_clients"] and r["acc_held"]:
            verdict = "R1+R2 使路由复活且准确率守住 —— 修复方向可行"
        elif r["n_keyH_revived"]:
            verdict = "R1+R2 使路由复活但准确率未守住 —— 需检查 R3/训练强度"
        else:
            verdict = "R1+R2 未能复活路由（τ 或微调强度可能不足）"
    else:
        verdict = "reset 臂无可评估客户端"
    return "；".join(parts) + "。" + verdict + "。"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    probe = parse_args()
    run_dir = Path(probe.run_dir).resolve()
    device = probe.device

    seed_everything(probe.seed)

    run_args = load_args_from_run_dir(run_dir, data_path=probe.data_path,
                                      device=device)
    if run_args.data_name == 'cifar100':
        run_args.nb_classes = 100
    elif run_args.data_name == 'ImageNet-R':
        run_args.nb_classes = 200
    else:
        raise ValueError(f"unsupported data_name: {run_args.data_name!r}")
    probe.num_workers = (probe.num_workers if probe.num_workers is not None
                         else run_args.num_workers)

    ckpt_path = Path(resolve_checkpoint(run_dir, probe.checkpoint))
    raw = load_checkpoint_file(str(ckpt_path))
    # raw is deleted after the arms loop (memory hygiene), so capture the
    # checkpoint's progress marker now for the JSON meta block.
    completed_round = raw.get("completed_round")
    inv = checkpoint_inventory(raw)
    print("[inventory] checkpoint keys:", ", ".join(inv["checkpoint_keys"]))
    print(f"[inventory] n_clients={inv['n_clients']} "
          f"data_split_state={inv['has_data_split_state']} "
          f"outer_split_manifest={inv['has_outer_split_manifest']} "
          f"pool{inv['prompt_pool_shape']} key{inv['prompt_key_shape']}")

    server = build_server_from_manifest(run_args, raw, device)
    pool_meta = {"pool_size": int(server.model.prompt.pool_size),
                 "top_k": int(server.model.prompt.top_k),
                 "prompt_length": int(server.model.prompt.length)}

    arms = ["keep", "reset"] if probe.arm == "both" else [probe.arm]
    arms_out = {}
    try:
        for arm_idx, arm in enumerate(arms):
            print(f"\n########## arm={arm} "
                  f"({'R1+R2' if arm == 'reset' else 'R2 only'}) "
                  f"tau={probe.tau} ##########")
            # deterministic per-(arm, client) fine-tune RNG streams
            seed_everything(FT_SEED_BASE + 1000 * (probe.seed % 1000)
                            + 100 * arm_idx)
            arms_out[arm] = run_arm(arm, arm_idx, server, raw, run_args,
                                    probe, device)
            print_arm_table(arm, arms_out[arm])
    finally:
        uninstall_soft_route()
        del raw

    summaries = {arm: arm_summary(arms_out[arm]) for arm in arms_out}
    iso = isolation_diff(arms_out)
    conclusion = build_conclusion(summaries)

    results = {
        "meta": {
            "script": "diagnostics/probe_route_fix.py",
            "probe": ("A2-R route-revival micro fine-tune probe: R1(orthonormal "
                      "key reset) + R2(differentiable soft routing), NO R3"),
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "data_name": run_args.data_name,
            "device": device,
            "client_num": run_args.client_num,
            "task_num": run_args.task_num,
            "nb_classes": run_args.nb_classes,
            "completed_round_of_checkpoint": completed_round,
            **pool_meta,
            "tau": probe.tau,
            "epochs": probe.epochs,
            "batch_size": probe.batch_size,
            "lr": probe.lr,
            "weight_decay": WEIGHT_DECAY,
            "pull_coeff": PULL_COEFF,
            "seed": probe.seed,
            "num_workers": probe.num_workers,
            "arm_semantics": {
                "reset": "R1 (prompt_key rows orthonormal reset) + R2 (soft routing)",
                "keep": "R2 only (checkpoint prompt_key kept)",
            },
            "patch": {
                "target": ("Models/global_prompt.py::Global_Prompt.forward "
                           "(runtime class-level swap; no Models/ file modified)"),
                "mechanism": ("w = softmax(sim/tau) over ALL pool slots; "
                              "batched_prompt = einsum(w, pool); token count "
                              "preserved by repeating the mixed prompt top_k "
                              "times (identity when top_k==1, as in this run)"),
                "coverage": ("vision_transformer__l2p.forward_features calls "
                             "self.prompt(...) on BOTH train and evaluate "
                             "paths -> patching Global_Prompt.forward covers "
                             "every route; batchwise majority vote bypassed "
                             "by per-sample soft weights; prompt_mask "
                             "fail-closed (use_prompt_mask=False in this run)"),
            },
            "loss": ("CE(masked logits, target) - 0.1 * soft reduce_sim; "
                     "soft reduce_sim = top_k * mean_b(sum_s w*sim), "
                     "scale-matched to the official hard reduce_sim "
                     "(sum of top-k cosines / B) -> official pull semantics "
                     "compatible, kept (no pure-CE fallback needed)"),
            "trainable": ("vit.prompt (pool + key) + per-task Chead heads[t]; "
                          "Tail_Anchor key/anchor_pool + ViT backbone frozen "
                          "(optimizer-scope freezing)"),
            "eval_semantics": ("official Client_DF.evaluate verbatim (batch=8 "
                               "shuffle=True, seed fixed per (client,task)); "
                               "before = hard routing, after = soft routing "
                               "(the probe's deployed form)"),
            "isolation": ("per-arm full re-restore from the in-memory raw "
                          "checkpoint (restore_boundary); fine-tune mutations "
                          "live on per-client deepcopies only; before baselines "
                          "re-measured per arm and cross-checked"),
            "thresholds": {
                "keyH_revival_bits": KEYH_REVIVAL_BITS,
                "mean_drop_tol_pp": MEAN_DROP_TOL_PP,
                "task_drop_tol_pp": TASK_DROP_TOL_PP,
            },
            "checkpoint_inventory": inv,
        },
        "arms": arms_out,
        "arm_summaries": summaries,
        "isolation_check": iso,
        "conclusion": conclusion,
    }

    print("\n[结论] " + conclusion)
    if iso is not None:
        print(f"[隔离检查] 两臂 before 完全一致 = "
              f"{iso['before_identical_across_arms']} "
              f"(max_abs_diff={iso['max_abs_diff']})")

    out_path = Path(probe.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
