"""A3 offline calibration: routing-consistency loss design parameters.

Pre-training calibration for method A3 (functional-level routing consistency
loss  L = L_task + lambda * KL(R_old || R_new),  R = softmax(tau * sim(z, K))).
Pure inference + one-step analytic gradients, zero training. Answers the
three unknowns before any A3 training code is written:

  T-a  temperature tau:
       Forward the SAME reference inputs (public surrogate data, deterministic
       per-client sample of --n_ref items, identical batch composition in both
       passes) through both generation interfaces, take the raw key-routing
       similarity matrices (B x pool_size, computed exactly like
       Tail_Anchor.forward: l2-normalized features . l2-normalized key rows,
       taken BEFORE anchor selection), sweep tau over --temps and report the
       top-1 probability and entropy (nats) of R = softmax(tau * sim).
       Target: the tau window where top-1 probability is ~[0.6, 0.9].

  T-b  constraint magnitude (the decisive numbers):
       PRIMARY per-sample KL on the FULL slot distribution, restricted to
       samples whose OLD top-1 destination is a protected slot:
           KL_prot = mean_{i: argmax R_old_i in P} KL(R_old_i || R_new_i).
       (The within-mask RENORMALIZED KL is also reported as
       `within_protected_kl`, but only as an auxiliary: renormalization
       erases exactly the protected->unprotected probability-mass escape
       that A3 must prevent, so it must NOT decide the loss form.)
       lambda calibration, loss-ratio AND gradient-ratio:
         lambda_loss ~ L_task_CE / KL_prot              (coarse reference)
         lambda_grad ~ ||grad_K L_base|| / ||grad_K KL||  (PRIMARY)
       The gradient ratio is primary because in the official phase-2 loss
       (CE + 0.2*G*InfoNCE - 0.1*pull, Adam wd=1e-3) the key receives NO
       continuous gradient from CE or InfoNCE (both reach the key only
       through the discrete topk index); the key's only continuous training
       signal is -0.1*pull + weight decay. The KL term therefore competes
       with || -0.1*grad_K pull + wd*K ||, not with the CE scalar.
       Both gradient norms are measured analytically (autograd w.r.t. the
       key only; features are constants w.r.t. the key). Two variants of
       the KL gradient are reported: DEPLOYMENT (feat_new: current drifted
       prompts — the lambda-relevant one, what A3's training actually sees)
       and ISOLATION (feat_old: prompts held at the old generation —
       comparing the two LAMBDAS only says whether the deployment lever is
       weaker, it does NOT attribute drift). Factor attribution is carried
       by a single-factor drift decomposition instead: key-only KL
       (features at the old generation, drifted key) vs prompt-only KL
       (drifted features, key at the old generation) — non-additive, but
       each term isolates one factor of the interface. lambda_grad is the
       CENTER of a sweep grid (scan at least +-1 order of magnitude), not
       an exact training-scale equivalence: Adam rescales per-parameter
       update magnitudes, so gradient-norm balance is a first-order
       estimate only.

  T-c  protection-set mask separation:
       top-1 flip rate (old-vs-new argmax disagreement, conditioned on the
       OLD interface's top-1 destination being a protected slot), stratified
       by the OLD routing margin (top1-top2 similarity, similarity space)
       — reported as STRUCTURE only. Note the two spaces are different: a
       large sim-space margin can coexist with a soft softmax distribution
       at small tau, so the margin is never used to infer softmax
       sharpness; sharpness evidence is the measured top-1 probability of
       the old routing distribution (T-a). Plus mean per-column JS distance
       (column j: the distribution over reference samples routed to slot j,
       old vs new), computed separately for protected and unprotected
       slots.

Interface definitions (aligned with compatibility_drift.py's 4-combo):
  old = (old prompt, old key, old anchor)          from --old_ckpt
  new = (current prompt, current key, current anchor) from --new_ckpt
Routing similarity depends only on the key and the features (the anchor is
selected AFTER routing), so anchors never enter the routing math; the new
anchor/head only matter for the task-loss forward.

Checkpoint pairs:
  lambda / loss-form decisions should use ADJACENT pairs
  (task_03_end -> task_04_end, one task of drift): A3 snapshots R_old at
  every task start, so the single-task drift window is the deployment
  condition, and at new_ckpt the KL is at its deployment-relevant maximum.
  Multi-task spans (task_02_end -> task_04_end, task_00_end -> task_04_end)
  are still useful as ACCUMULATED-DRIFT diagnosis, but their KL is not the
  number lambda should balance.

Protection set: ALWAYS replicated from the A2 mechanism at the A3 snapshot
moment (= the start of the task FOLLOWING old_ckpt): top-1 routing of ALL
completed tasks' (0..old_task INCLUSIVE) train-split features under the OLD
key, union of hit slots. Mirrors Client_DF._a2_init_task_protection exactly
at that moment: its `completed` is range(task_id) = range(old_task+1), its
"current key at task start" is exactly the old_ckpt key, its ViT task_id is
the FOLLOWING task (old_task + 1), and its loader batch size is the
hardcoded official 16 (A2_PROTECTION_BATCH below; batchwise prompt voting
makes features batch-composition dependent, so replication must not vary
it). RNG-neutral: sequential loader, shuffle=False, num_workers=0, eval
mode. The a2_state stored in old_ckpt is deliberately NOT used: it was
computed at the START of old_ckpt's own task (tasks 0..old_task-1, older
key) — one task stale relative to the A3 snapshot moment. Its presence is
only recorded.

RNG discipline (same as compatibility_drift.py): the whole diagnostic runs
under an RNG snapshot that is restored at the end; both interface passes are
seeded with the fixed per-(client, task) convention 9700 + 100*cid + task;
per-client reference/batch sampling uses detached torch.Generator objects so
the global stream is never consumed; the current vit prompts are reloaded
into the shared vit after every old-interface pass.

Outputs:
  CSV  (default <run_dir>/metrics/a3_calibration.csv): one row per
       (client x tau): top1/entropy stats, the three KL flavors, gradient
       norms + both lambda ratios, flip rates (overall + margin-stratified),
       old protected-sample margins, per-column JS.
  JSON (same directory, .csv suffix -> .json): metadata, tau sweep summary,
       task-loss reference, gradient calibration, and the three verdicts.
  Console: per-client lines + the three-part verdict summary.

Usage (repo root, on the server; ADJACENT pair for lambda decisions):
  python diagnostics/a3_offline_calibration.py \
      --run_dir output/ImageNet-R/fedta/inr_base_p40/seed_42 \
      --old_ckpt checkpoints/task_03_end.pth \
      --new_ckpt checkpoints/task_04_end.pth \
      --n_ref 256
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server, resolve_checkpoint
from runtime_utils import (load_checkpoint_file, clone_prompt_template,
                           rng_state_dict, restore_rng_state)

# fixed per-(client, task) seeding convention (compatibility_drift.py)
SEED_BASE = 9700

# soft reference threshold only: a KL below it does NOT by itself decide
# "KL ineffective" — the verdict jointly considers flip rate, routing
# margins, and the analytic KL gradient norm (see make_kl_verdict)
KL_SOFT_FLOOR = 0.01

# T-a target window for softmax top-1 probability
TOP1_WINDOW = (0.6, 0.9)
TOP1_TARGET = 0.75

# exact official constants (do NOT expose as CLI flags: changing them would
# break the "exact replica" claim of this diagnostic)
A2_PROTECTION_BATCH = 16   # Client_DF._a2_init_task_protection hardcodes 16
TRAIN_BATCH = 16           # official train() DataLoader hardcodes batch_size=16
PULL_COEF = 0.1            # phase-2 loss: CE + 0.2*G*InfoNCE - 0.1*pull
WEIGHT_DECAY = 1e-3        # Adam(self.model.parameters(), weight_decay=1e-3)


def parse_args():
    p = argparse.ArgumentParser("A3 offline calibration (tau / KL / mask)")
    p.add_argument("--run_dir", required=True,
                   help=".../output/{data}/{method}/{run}/seed_{seed}")
    p.add_argument("--old_ckpt", required=True,
                   help="checkpoint at the END of the old task "
                        "(relative to cwd, run_dir, or absolute)")
    p.add_argument("--new_ckpt", default=None,
                   help="current checkpoint (default: checkpoints/latest.pth)")
    p.add_argument("--old_task", type=int, default=None,
                   help="override the old task id (default: task id stored "
                        "in old_ckpt)")
    p.add_argument("--n_ref", type=int, default=256,
                   help="reference inputs per client (public surrogate data)")
    p.add_argument("--ref_batch_size", type=int, default=16,
                   help="batch size for the two reference-input interface "
                        "passes (identical composition in both passes, so "
                        "batchwise prompt voting cancels)")
    p.add_argument("--temps", default="1,2,5,10,20,50",
                   help="comma-separated tau sweep for R = softmax(tau*sim)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_csv", default=None,
                   help="default <run_dir>/metrics/a3_calibration.csv")
    p.add_argument("--seed", type=int, default=None,
                   help="global seed for the diagnostic (default: run seed)")
    p.add_argument("--data_path", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# exact replicas of the training-time math (observer-only; no model files
# are touched — Tail_Anchor.forward hardcodes .to('cuda'), so the routing
# similarity is recomputed here device-agnostically)
# ---------------------------------------------------------------------------

def l2_normalize(x, dim=1, epsilon=1e-12):
    """Exact replica of Tail_Anchor.l2_normalize."""
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(
        square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm


def routing_similarity(feat, key):
    """Tail_Anchor routing similarity, BEFORE anchor selection:
    l2_normalize(feat) @ l2_normalize(key rows).t()  ->  (N, pool_size)."""
    x_norm = l2_normalize(feat, dim=1)
    key = key.reshape(-1, key.shape[-1])
    key_norm = l2_normalize(key, dim=1)
    return torch.matmul(x_norm, key_norm.t())


def forward_features(client, x, task_id, batch_size):
    """No-grad features of a fixed input tensor under whatever prompts are
    currently loaded in the shared vit. Eval mode is assumed (set once in
    main). Mirrors evaluate()/_a2_init_task_protection: vit forward with
    task_id=<current task>, train=True (use_prompt_mask=False in this
    codebase, so the flag is inert)."""
    feats = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(client.device, non_blocking=True)
        with torch.no_grad():
            output = client.original_model(xb)
            cls_features = output['pre_logits']
            output = client.vit(xb, task_id=task_id,
                                cls_features=cls_features, train=True)
        feats.append(output['feat'].to(client.device))
    return torch.cat(feats, dim=0)


def interface_features(client, prompt_module, ref_x, task_id, batch_size):
    """Features of the fixed reference tensor under one interface.
    The shared vit gets the interface's prompts loaded (the new-interface
    pass that follows reloads the current ones)."""
    client.vit.load_prompts(prompt_module)
    return forward_features(client, ref_x, task_id, batch_size)


def get_train_split(client, task_k):
    """data_split_indices keys may be int (fresh) or str (post-serialize)."""
    dsi = client.data_split_indices
    if task_k in dsi:
        return dsi[task_k]["train"]
    if str(task_k) in dsi:
        return dsi[str(task_k)]["train"]
    raise KeyError(f"data_split_indices missing task {task_k}")


def replicate_protection(client, old_prompt_module, old_key, old_task):
    """A2 protection-set replication at the A3 snapshot moment: union over
    completed tasks 0..old_task of the top-1 routing destinations of their
    train-split features under the OLD key — exactly what
    Client_DF._a2_init_task_protection computes when the task FOLLOWING
    old_ckpt starts (its completed range is range(old_task+1), its ViT
    task_id is old_task+1, its key is the task-start key = the old_ckpt
    key, its loader batch size is the hardcoded 16). RNG-neutral: sequential
    loader, shuffle=False, num_workers=0, eval mode."""
    key = old_key.to(client.device)
    snapshot_task = old_task + 1          # the task A3 would start training
    client.vit.load_prompts(old_prompt_module)
    hit_slots = []
    for t in range(0, old_task + 1):
        dataset = Subset(client.train_data[t], get_train_split(client, t))
        loader = DataLoader(dataset, batch_size=A2_PROTECTION_BATCH,
                            num_workers=0, shuffle=False)
        feats = []
        for input, _target in loader:
            input = input.to(client.device, non_blocking=True)
            with torch.no_grad():
                output = client.original_model(input)
                cls_features = output['pre_logits']
                output = client.vit(input, task_id=snapshot_task,
                                    cls_features=cls_features, train=True)
            feats.append(output['feat'].to(client.device))
        if not feats:
            continue
        feat = torch.cat(feats, dim=0)
        similarity = routing_similarity(feat, key)
        _, index = torch.topk(similarity, k=1)
        hit_slots.append(index.reshape(-1).cpu())
    if not hit_slots:
        return torch.zeros(0, dtype=torch.long)
    return torch.unique(torch.cat(hit_slots))


def deterministic_train_batch(client, gen):
    """One deterministic batch (TRAIN_BATCH, official train() size) of the
    current task's train data, as (x, y) tensors on CPU."""
    n = len(client.traindata)
    take = min(TRAIN_BATCH, n)
    idx = torch.randperm(n, generator=gen)[:take].tolist()
    xs, ys = [], []
    for i in idx:
        x, y = client.traindata[i]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def task_loss_ce(client, feat_cur, target, nb_classes):
    """Official train() phase-2 cross-entropy on precomputed features:
    Tail_Anchor anchor-top-1 + current head, classes outside current_class
    masked to -inf over the first nb_classes logits (exact train() replica;
    pull/InfoNCE terms are handled separately by the gradient calibration)."""
    with torch.no_grad():
        similarity = routing_similarity(feat_cur, client.model.key.detach())
        _, index = torch.topk(similarity, k=1)
        anchor = client.model.anchor_pool[index].reshape(
            -1, feat_cur.shape[1])
        x1 = torch.stack((feat_cur, anchor), dim=1).reshape(
            -1, feat_cur.shape[1] * 2)
        logits = client.model.head(x1)
        not_mask = np.setdiff1d(np.arange(nb_classes), client.current_class)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(client.device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        return float(F.cross_entropy(logits, target))


def kl_key_grad(feat, key, log_ro, tau, sample_mask=None):
    """||grad_K KL(R_old || R_new)|| on reference features, analytic
    (autograd w.r.t. the key only; features are constants w.r.t. the key).
    log_ro: detached OLD-interface log-routing (the fixed target).
    sample_mask: optional bool mask over samples — when given, the KL is
    averaged over the masked subset only (matches the sample-masked A3
    loss form on protected samples).

    Two semantic variants are measured in main():
      deployment gradient  (feat_new, K_new): A3's actual training-time
        lever — with the CURRENT (drifted) prompts fixed, how much key
        gradient the constraint produces (this is the lambda-relevant one);
      isolation gradient   (feat_old, K_new): if prompts had NOT drifted,
        how much key gradient pure key-repair would need — comparing the
        two decomposes the constraint's difficulty into the prompt-drift
        and key-drift components."""
    key_g = key.detach().clone().requires_grad_(True)
    sim = routing_similarity(feat, key_g)
    log_rn = F.log_softmax(tau * sim, dim=1)
    per_sample = (log_ro.exp() * (log_ro - log_rn)).sum(dim=1)
    if sample_mask is None:
        kl = per_sample.mean()
    else:
        if not bool(sample_mask.any()):
            del key_g, sim, log_rn, per_sample
            return float('nan')
        kl = per_sample[sample_mask].mean()
    g, = torch.autograd.grad(kl, key_g)
    out = float(g.norm())
    del key_g, sim, log_rn, per_sample, kl, g
    return out


def kl_key_rowgrad(feat, key, log_ro, tau, sample_mask=None):
    """Per-ROW gradient norms of the (sample-masked) KL w.r.t. the key,
    shape (n_slots,). Diagnoses WHERE the large KL gradient concentrates:
    if a few key rows carry most of the total gradient norm, lambda_grad
    is dominated by parameter-scale structure rather than by drift
    structure. Returns None when the sample mask is empty."""
    key_g = key.detach().clone().requires_grad_(True)
    sim = routing_similarity(feat, key_g)
    log_rn = F.log_softmax(tau * sim, dim=1)
    per_sample = (log_ro.exp() * (log_ro - log_rn)).sum(dim=1)
    if sample_mask is None:
        kl = per_sample.mean()
    else:
        if not bool(sample_mask.any()):
            del key_g, sim, log_rn, per_sample
            return None
        kl = per_sample[sample_mask].mean()
    g, = torch.autograd.grad(kl, key_g)
    out = g.reshape(-1, g.shape[-1]).norm(dim=1).detach().clone()
    del key_g, sim, log_rn, per_sample, kl, g
    return out


def pull_key_grad(feat, key):
    """grad_K of the official pull term (reduce_sim), exact replica of
    Tail_Anchor.forward's pull computation on `feat` under `key`:
    reduce_sim = sum(l2norm(key)[top1] * l2norm(feat)) / key_size.

    Scale VERIFIED against Models/Tail_Anchor.py (L61-64): the official
    reduce_sim = torch.sum(sim) / self.key_size is a SUM over
    B * top_k * C elements divided by the feature dim (768) — NOT a batch
    mean — so this gradient scales linearly with the batch size, exactly
    as in official training (train() hardcodes batch_size=16 = TRAIN_BATCH,
    which is what this replica is called with; the gradient-norm ratio is
    therefore at deployment scale, no B-fold bias). The integer-index
    gather key_norm[index] is differentiable w.r.t. the selected rows
    (PyTorch gather semantics; `index` itself needs no grad) and is the
    same operation the official forward performs.

    Returns the gradient tensor (caller combines it with weight decay).
    This is the ONLY continuous channel through which the official phase-2
    loss trains the key — CE and InfoNCE reach the key only via the
    discrete topk index, which passes no gradient."""
    key_g = key.detach().clone().requires_grad_(True)
    key_norm = l2_normalize(key_g.reshape(-1, key_g.shape[-1]), dim=1)
    x_norm = l2_normalize(feat, dim=1)
    sim = torch.matmul(x_norm, key_norm.t())
    _, index = torch.topk(sim, k=1)
    batched_key_norm = key_norm[index]           # B, top_k, C
    x_embed_norm = x_norm.unsqueeze(1)           # B, 1, C
    reduce_sim = torch.sum(batched_key_norm * x_embed_norm) / key_g.shape[-1]
    g, = torch.autograd.grad(reduce_sim, key_g)
    del key_g, key_norm, sim, index, batched_key_norm, x_embed_norm, reduce_sim
    return g


def within_masked_kl(log_rold, log_rnew, cols):
    """AUXILIARY metric (do NOT use for loss-form decisions): per-sample
    KL(R_old || R_new) restricted to the protection-set columns, both
    distributions RENORMALIZED within the mask. Renormalization erases the
    protected->unprotected mass escape, so this number can be ~0 while
    routing is catastrophically broken; it only measures redistribution
    AMONG protected slots."""
    lo = log_rold[:, cols] - torch.logsumexp(log_rold[:, cols], dim=1,
                                             keepdim=True)
    ln = log_rnew[:, cols] - torch.logsumexp(log_rnew[:, cols], dim=1,
                                             keepdim=True)
    per_sample = (lo.exp() * (lo - ln)).sum(dim=1)
    return float(per_sample.mean())


def column_js_distance(R_old, R_new, eps=1e-12):
    """Per-slot JS distance between the two interfaces' slot-usage profiles.
    Column j: profile = R[:, j] normalized over reference samples;
    JS = sqrt(JSD(profile_old_j, profile_new_j)) with 0*log(0)=0."""
    P = R_old.shape[1]
    out = torch.zeros(P, dtype=torch.float64, device=R_old.device)
    for j in range(P):
        p = R_old[:, j].double()
        q = R_new[:, j].double()
        mp, mq = p.sum(), q.sum()
        if mp < eps and mq < eps:
            continue  # no mass on either side: no drift
        pp = p / mp if mp >= eps else torch.zeros_like(p)
        qq = q / mq if mq >= eps else torch.zeros_like(q)
        m = 0.5 * (pp + qq)

        def kl_terms(a):
            mask = a > 0
            return (a[mask] * (torch.log(a[mask]) - torch.log(m[mask]))).sum()

        jsd = 0.5 * kl_terms(pp) + 0.5 * kl_terms(qq)
        out[j] = torch.sqrt(torch.clamp(jsd, min=0.0))
    return out


def make_kl_verdict(psk, old_top1, kl_grad, base_grad, lam_grad, prot_flip):
    """DESCRIPTIVE joint verdict — no auto-selection of the loss form.

    A small protected-sample KL alone decides nothing:
      (a) top-1 flips can be CHEAP in probability space when the old
          routing distribution is soft — the sharpness evidence is the
          MEASURED old top-1 probability, never the sim-space top1-top2
          margin (a large sim margin can coexist with a soft softmax at
          small tau; they live in different spaces);
      (b) a weak constraint lever shows up as a small analytic KL key
          gradient, which a larger lambda/tau can compensate.
    The verdict therefore reports the measured structure and names the
    viable constraint forms instead of auto-selecting one."""
    if not np.isfinite(psk):
        return ("INDETERMINATE: no protected samples (empty protection "
                "set or no protected top-1 destinations)", None)
    if psk < KL_SOFT_FLOOR and prot_flip < 0.01:
        return (f"NO DRIFT on protected samples at this snapshot "
                f"(KL={psk:.4f} < {KL_SOFT_FLOOR}, flip rate "
                f"{prot_flip:.3f} < 0.01): the constraint is inactive — "
                f"there is nothing to preserve here; do NOT add a hinge "
                f"either (zero drift means zero constraint need)", False)
    parts = []
    soft = old_top1 < 0.8   # measured softmax sharpness at rec tau
    if psk >= KL_SOFT_FLOOR:
        parts.append(
            f"protected-sample full KL = {psk:.4f} >= {KL_SOFT_FLOOR}: "
            f"measurable distribution drift on the samples A3 must protect "
            f"-> the soft-KL constraint has signal to act on")
        effective = True
    elif soft:
        parts.append(
            f"protected-sample full KL = {psk:.4f} < {KL_SOFT_FLOOR} with a "
            f"SOFT old routing distribution (measured old top-1 prob = "
            f"{old_top1:.3f}, protected flip rate {prot_flip:.3f}): top-1 "
            f"flips are cheap in probability space, so a small KL does NOT "
            f"mean no damage — it means a soft-KL term must be driven "
            f"harder (larger tau/lambda; check lambda_grad below), while a "
            f"winner-preserving margin hinge would enforce the top-1 "
            f"directly in similarity space, immune to softmax softness")
        effective = None
    else:
        parts.append(
            f"protected-sample full KL = {psk:.4f} < {KL_SOFT_FLOOR} with a "
            f"SHARP old routing distribution (measured old top-1 prob = "
            f"{old_top1:.3f}, protected flip rate {prot_flip:.3f}): little "
            f"distributional difference for a soft KL to grip — the "
            f"margin-hinge form targets this regime directly")
        effective = None
    if np.isfinite(kl_grad) and kl_grad > 0 and np.isfinite(base_grad):
        parts.append(
            f"gradient calibration: ||grad_K L_base|| = {base_grad:.3e} vs "
            f"||grad_K KL_prot|| = {kl_grad:.3e} -> lambda_grad ~ "
            f"{lam_grad:.2e} as the SWEEP CENTER, not an exact equivalence "
            f"(Adam rescales per-parameter updates; scan at least +-1 order "
            f"of magnitude around it; the CE scalar passes no continuous "
            f"gradient to the key)")
    else:
        parts.append(
            "gradient calibration unavailable (KL key-gradient ~ 0 at the "
            "recommended tau): a soft-KL term would have no lever on the "
            "key — use the margin-hinge form")
    parts.append(
        "margin-hinge reference form (winner-preserving, NOT similarity "
        "regression): L_margin = max(0, m - s_new(x, k_old*) + "
        "max_{j != old*} s_new(x, k_j)) with old* = argmax_j s_old(x, k_j) "
        "— the slot the OLD interface chose must stay ahead of every "
        "competitor by at least m under the NEW interface")
    return ("; ".join(parts), effective)


def main():
    args = parse_args()
    old_path = resolve_checkpoint(args.run_dir, args.old_ckpt)
    new_path = (resolve_checkpoint(args.run_dir, args.new_ckpt)
                if args.new_ckpt
                else resolve_checkpoint(args.run_dir, "checkpoints/latest.pth"))
    temps = [float(t) for t in args.temps.split(",") if t.strip()]

    # environment from the NEW checkpoint (current prompts / key / anchor /
    # heads / test loaders all restored by bootstrap_server)
    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=new_path,
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes
    method = getattr(run_args, "method", None)
    a2_arm = getattr(run_args, "a2_arm", None)
    a2_run_drift_suppressed = (method == 'fedta_a2'
                               and a2_arm in ('key', 'both'))
    if a2_run_drift_suppressed:
        print("!" * 78)
        print(f"[A2-RUN WARNING] this run is method fedta_a2 (arm={a2_arm}):")
        print("protected key rows are FROZEN during training, so ALL drift")
        print("numbers (protected-sample KL, flip rates) are SUPPRESSED and")
        print("the recommended lambda is NOT deployment-valid — rerun this")
        print("calibration on a pure fedta baseline run before deciding")
        print("the A3 loss form or lambda")
        print("!" * 78)

    old_ckpt = load_checkpoint_file(old_path)

    # ---- public surrogate reference pool (kd_fusion_prompt protocol) ----
    surro = server.surrogate_data
    classes = sorted({int(t) for t in surro.targets})
    surro.getTrainData(classes)
    n_pool = len(surro)
    n_ref = min(args.n_ref, n_pool)

    seed = args.seed if args.seed is not None else run_args.seed
    print(f"[a3-calib] old_ckpt={old_path}")
    print(f"[a3-calib] new_ckpt={new_path}")
    print(f"[a3-calib] surrogate pool={n_pool} samples over {len(classes)} "
          f"classes; n_ref={n_ref}/client; temps={temps}; seed={seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rng_snapshot = rng_state_dict()
    # the vit and the frozen original model are SHARED across clients:
    # snapshot their modes once, eval() for all feature passes, restore after
    vit_was_training = server.model.training
    orig_was_training = server.origin_model.training

    rows = []
    per_client = {}
    rowgrad_detail = {}
    n_skipped = 0
    try:
        server.model.eval()
        server.origin_model.eval()
        for client in server.clients:
            cid = client.id
            old_state = old_ckpt["clients"][cid]
            old_task = (args.old_task if args.old_task is not None
                        else old_state["task_id"])
            new_task = client.task_id
            if old_task is None or old_task < 0:
                print(f"client {cid}: no old task in old_ckpt, skipped")
                n_skipped += 1
                continue
            if new_task is None or new_task < 0 or new_task <= old_task:
                print(f"client {cid}: old task {old_task} not before current "
                      f"task {new_task}, skipped")
                n_skipped += 1
                continue
            if old_state.get("prompts") is None or client.prompts is None:
                print(f"client {cid}: prompts missing in one checkpoint, "
                      f"skipped")
                n_skipped += 1
                continue
            if len(client.traindata or []) == 0:
                print(f"client {cid}: current train split empty, skipped")
                n_skipped += 1
                continue

            cur_prompt = client.prompts
            old_prompt = clone_prompt_template(client.vit)
            old_prompt.load_state_dict(old_state["prompts"])
            old_key = old_state["tail_anchor_model"]["key"].detach().clone()
            client.model.to(client.device)
            cur_key = client.model.key.detach().clone()
            pool_size = cur_key.shape[0]

            # ---- protection set: ALWAYS replicate at the A3 snapshot moment
            # (a2_state in old_ckpt is one task STALE — computed at the start
            # of old_ckpt's own task; only its presence is recorded)
            a2 = old_state.get("a2_state")
            torch.manual_seed(SEED_BASE + 100 * cid + old_task)
            slots = replicate_protection(client, old_prompt, old_key,
                                          old_task)
            prot_cols = sorted(int(s) for s in slots.tolist())
            prot_src = (f"replicated A2 mechanism (old key over train "
                        f"splits of tasks 0..{old_task}, ViT task_id="
                        f"{old_task + 1}, batch={A2_PROTECTION_BATCH})")
            prot_set = set(prot_cols)
            unprot_cols = [j for j in range(pool_size) if j not in prot_set]

            # ---- fixed reference inputs (identical for both interfaces) --
            gen = torch.Generator()
            gen.manual_seed(SEED_BASE + 100 * cid + new_task)
            perm = torch.randperm(n_pool, generator=gen)[:n_ref]
            ref_x = torch.stack([surro[int(i)][1] for i in perm.tolist()])
            n_actual = int(ref_x.shape[0])

            # ---- old interface pass ----
            torch.manual_seed(SEED_BASE + 100 * cid + new_task)
            feat_old = interface_features(client, old_prompt, ref_x, new_task,
                                          args.ref_batch_size)
            sim_old = routing_similarity(feat_old, old_key.to(client.device))

            # ---- new interface pass (current prompts restored first) ----
            torch.manual_seed(SEED_BASE + 100 * cid + new_task)
            feat_new = interface_features(client, cur_prompt, ref_x, new_task,
                                          args.ref_batch_size)
            sim_new = routing_similarity(feat_new, cur_key.to(client.device))

            # ---- current-task batch: CE magnitude + pull gradient (new
            # interface; features are constants w.r.t. the key) ----
            gen2 = torch.Generator()
            gen2.manual_seed(SEED_BASE + 100 * cid + new_task + 1)
            batch_x, batch_y = deterministic_train_batch(client, gen2)
            batch_y = batch_y.to(client.device)
            feat_cur = forward_features(client, batch_x, new_task,
                                        args.ref_batch_size)
            task_loss = task_loss_ce(client, feat_cur, batch_y, nb_classes)
            g_pull = pull_key_grad(feat_cur, cur_key.to(client.device))
            # official instantaneous base gradient on the key:
            # d/dK [ -0.1 * pull + wd * K ]  (Adam L2 adds wd*K to the grad)
            g_base = (-PULL_COEF * g_pull
                      + WEIGHT_DECAY * cur_key.reshape(-1, cur_key.shape[-1]))
            base_grad_norm = float(g_base.norm())
            pull_grad_norm = float(g_pull.norm())

            # ---- T-c flip rates + old routing margins (tau-independent) --
            argm_old = sim_old.argmax(dim=1)
            argm_new = sim_new.argmax(dim=1)
            flip = argm_old != argm_new
            prot_col_mask = torch.zeros(pool_size, dtype=torch.bool,
                                        device=sim_old.device)
            if prot_cols:
                prot_col_mask[torch.tensor(prot_cols, dtype=torch.long,
                                           device=sim_old.device)] = True
            sample_prot = prot_col_mask[argm_old]
            prot_flip = (float(flip[sample_prot].float().mean())
                         if bool(sample_prot.any()) else float('nan'))
            unprot_flip = (float(flip[~sample_prot].float().mean())
                           if bool((~sample_prot).any()) else float('nan'))

            # old routing margin (top1 - top2 similarity, similarity space)
            top2 = torch.topk(sim_old, k=2).values
            margin_old = top2[:, 0] - top2[:, 1]
            prot_margins = margin_old[sample_prot]
            if int(sample_prot.sum()) >= 2:
                m_mean = float(prot_margins.mean())
                m_med = float(prot_margins.median())
                m_p10 = float(torch.quantile(prot_margins, 0.1))
                small = prot_margins < m_med
                flips_p = flip[sample_prot]
                flip_small = (float(flips_p[small].float().mean())
                              if bool(small.any()) else float('nan'))
                flip_large = (float(flips_p[~small].float().mean())
                              if bool((~small).any()) else float('nan'))
            else:
                m_mean = m_med = m_p10 = float('nan')
                flip_small = flip_large = float('nan')

            # key row norms (parameter-scale context for the gradient
            # norms: l2_normalize in routing_similarity means gradients
            # w.r.t. low-norm rows are amplified by ~1/||row||)
            kn_old = old_key.detach().reshape(
                -1, old_key.shape[-1]).norm(dim=1).float().cpu()
            kn_new = cur_key.detach().reshape(
                -1, cur_key.shape[-1]).norm(dim=1).float().cpu()
            prot_row_mask_c = torch.zeros(kn_new.numel(), dtype=torch.bool)
            prot_row_mask_c[torch.as_tensor(prot_cols)] = True
            kn_new_prot_mean = (float(kn_new[prot_row_mask_c].mean())
                                if bool(prot_row_mask_c.any())
                                else float('nan'))
            kn_new_unprot_mean = (float(kn_new[~prot_row_mask_c].mean())
                                  if bool((~prot_row_mask_c).any())
                                  else float('nan'))
            kn_new_min = float(kn_new.min())
            kn_old_min = float(kn_old.min())
            n_below_floor = int((kn_new < 1e-6).sum()
                                + (kn_old < 1e-6).sum())

            # ---- T-a / T-b per tau ----
            for tau in temps:
                log_ro = F.log_softmax(tau * sim_old, dim=1)
                log_rn = F.log_softmax(tau * sim_new, dim=1)
                ro, rn = log_ro.exp(), log_rn.exp()
                top1_o = float(ro.max(dim=1).values.mean())
                top1_n = float(rn.max(dim=1).values.mean())
                top1_o_prot = (float(ro[sample_prot]
                                     .max(dim=1).values.mean())
                               if bool(sample_prot.any())
                               else float('nan'))
                h_o = float((-(ro * log_ro).sum(dim=1)).mean())
                h_n = float((-(rn * log_rn).sum(dim=1)).mean())
                # PRIMARY KL: full distribution, protected SAMPLES only
                per_kl = (ro * (log_ro - log_rn)).sum(dim=1)
                prot_sample_kl = (float(per_kl[sample_prot].mean())
                                  if bool(sample_prot.any())
                                  else float('nan'))
                unprot_sample_kl = (float(per_kl[~sample_prot].mean())
                                    if bool((~sample_prot).any())
                                    else float('nan'))
                full_kl = float(per_kl.mean())
                # single-factor drift decomposition (non-additive, but each
                # term isolates one drifted factor of the interface):
                # prompt-only = features at new generation, key held old;
                # key-only    = features at old generation, drifted key
                sim_po = routing_similarity(feat_new,
                                            old_key.to(client.device))
                per_po = (ro * (log_ro - F.log_softmax(
                    tau * sim_po, dim=1))).sum(dim=1)
                sim_ko = routing_similarity(feat_old,
                                            cur_key.to(client.device))
                per_ko = (ro * (log_ro - F.log_softmax(
                    tau * sim_ko, dim=1))).sum(dim=1)
                prompt_only_kl = float(per_po.mean())
                key_only_kl = float(per_ko.mean())
                prompt_only_kl_prot = (float(per_po[sample_prot].mean())
                                       if bool(sample_prot.any())
                                       else float('nan'))
                key_only_kl_prot = (float(per_ko[sample_prot].mean())
                                    if bool(sample_prot.any())
                                    else float('nan'))
                # AUXILIARY: within-mask renormalized KL
                if prot_cols:
                    w_kl = within_masked_kl(log_ro, log_rn, prot_cols)
                    js_cols = column_js_distance(ro, rn)
                    p_js = float(js_cols[prot_cols].mean())
                else:
                    w_kl, p_js = float('nan'), float('nan')
                u_js = (float(column_js_distance(ro, rn)[unprot_cols].mean())
                        if unprot_cols else float('nan'))
                # analytic key gradients at this tau
                kl_grad_prot = kl_key_grad(feat_new,
                                           cur_key.to(client.device),
                                           log_ro, tau,
                                           sample_mask=sample_prot)
                kl_grad_all = kl_key_grad(feat_new,
                                          cur_key.to(client.device),
                                          log_ro, tau)
                # isolation variant: features held at the OLD generation
                # (prompts undrifted) — pure key-repair gradient
                kl_grad_prot_oldfeat = kl_key_grad(feat_old,
                                                   cur_key.to(client.device),
                                                   log_ro, tau,
                                                   sample_mask=sample_prot)
                # per-row gradient concentration at this tau (where does
                # the huge KL gradient actually live?)
                row_g = kl_key_rowgrad(feat_new,
                                       cur_key.to(client.device),
                                       log_ro, tau, sample_mask=sample_prot)
                if row_g is not None and float(row_g.sum()) > 0:
                    rg_total = float(row_g.sum())
                    top10_share = float(
                        row_g.topk(min(10, row_g.numel()))
                        .values.sum()) / rg_total
                    prot_row_mask = torch.zeros(row_g.numel(),
                                                dtype=torch.bool,
                                                device=row_g.device)
                    prot_row_mask[torch.as_tensor(
                        prot_cols, device=row_g.device)] = True
                    prot_rows_share = (float(row_g[prot_row_mask].sum())
                                       / rg_total)
                    # squared-norm shares — the statistic aligned with
                    # lambda's L2 denominator (one huge row can dominate
                    # the L2 norm while its L1 mass share looks modest)
                    rg_sq_total = float((row_g ** 2).sum())
                    top10_sq_share = float(
                        row_g.topk(min(10, row_g.numel()))
                        .values.pow(2).sum()) / rg_sq_total
                    prot_rows_sq_share = (float(
                        (row_g[prot_row_mask] ** 2).sum()) / rg_sq_total)
                else:
                    top10_share = prot_rows_share = float('nan')
                    top10_sq_share = prot_rows_sq_share = float('nan')
                # keep per-client row-grad detail for the recommended tau
                if row_g is not None:
                    rowgrad_detail[(cid, tau)] = {
                        "row_g": row_g.float().cpu(),
                        "prot_mask": prot_row_mask_c.clone(),
                        "kn_new": kn_new.clone(),
                    }
                lam_grad = (base_grad_norm / kl_grad_prot
                            if np.isfinite(kl_grad_prot) and kl_grad_prot > 0
                            else float('nan'))
                lam_loss = (task_loss / prot_sample_kl
                            if np.isfinite(prot_sample_kl)
                            and prot_sample_kl > 0 else float('nan'))
                rows.append({
                    "client_id": cid,
                    "old_task": old_task,
                    "new_task": new_task,
                    "temp": tau,
                    "top1_prob_mean_old": f"{top1_o:.6f}",
                    "top1_prob_old_prot": (f"{top1_o_prot:.6f}"
                                           if np.isfinite(top1_o_prot)
                                           else "nan"),
                    "top1_prob_mean_new": f"{top1_n:.6f}",
                    "top1_prob_mean": f"{(top1_o + top1_n) / 2:.6f}",
                    "entropy_mean_old": f"{h_o:.6f}",
                    "entropy_mean_new": f"{h_n:.6f}",
                    "entropy_mean": f"{(h_o + h_n) / 2:.6f}",
                    "protected_sample_kl": f"{prot_sample_kl:.6f}",
                    "unprotected_sample_kl": f"{unprot_sample_kl:.6f}",
                    "within_protected_kl": f"{w_kl:.6f}",
                    "full_kl": f"{full_kl:.6f}",
                    "prompt_only_kl": f"{prompt_only_kl:.6f}",
                    "prompt_only_kl_prot": f"{prompt_only_kl_prot:.6f}",
                    "key_only_kl": f"{key_only_kl:.6f}",
                    "key_only_kl_prot": f"{key_only_kl_prot:.6f}",
                    "base_grad_norm": f"{base_grad_norm:.6e}",
                    "pull_grad_norm": f"{pull_grad_norm:.6e}",
                    "kl_grad_norm_prot": (f"{kl_grad_prot:.6e}"
                                          if np.isfinite(kl_grad_prot)
                                          else "nan"),
                    "kl_grad_norm_prot_featold": (
                        f"{kl_grad_prot_oldfeat:.6e}"
                        if np.isfinite(kl_grad_prot_oldfeat) else "nan"),
                    "kl_grad_norm_all": (f"{kl_grad_all:.6e}"
                                         if np.isfinite(kl_grad_all)
                                         else "nan"),
                    "lambda_grad": (f"{lam_grad:.6e}"
                                    if np.isfinite(lam_grad) else "nan"),
                    "lambda_loss": (f"{lam_loss:.6e}"
                                    if np.isfinite(lam_loss) else "nan"),
                    "kl_rowgrad_top10_share": (
                        f"{top10_share:.6f}"
                        if np.isfinite(top10_share) else "nan"),
                    "kl_rowgrad_prot_rows_share": (
                        f"{prot_rows_share:.6f}"
                        if np.isfinite(prot_rows_share) else "nan"),
                    "kl_rowgrad_top10_sq_share": (
                        f"{top10_sq_share:.6f}"
                        if np.isfinite(top10_sq_share) else "nan"),
                    "kl_rowgrad_prot_rows_sq_share": (
                        f"{prot_rows_sq_share:.6f}"
                        if np.isfinite(prot_rows_sq_share) else "nan"),
                    "protected_flip_rate": f"{prot_flip:.4f}",
                    "unprotected_flip_rate": f"{unprot_flip:.4f}",
                    "flip_rate_small_margin": f"{flip_small:.4f}",
                    "flip_rate_large_margin": f"{flip_large:.4f}",
                    "old_margin_mean_prot": f"{m_mean:.6f}",
                    "old_margin_median_prot": f"{m_med:.6f}",
                    "old_margin_p10_prot": f"{m_p10:.6f}",
                    "protected_js": f"{p_js:.6f}",
                    "unprotected_js": f"{u_js:.6f}",
                    "n_ref": n_actual,
                    "n_protected_slots": len(prot_cols),
                    "n_unprotected_slots": len(unprot_cols),
                    "task_loss": f"{task_loss:.6f}",
                })

            per_client[cid] = {
                "old_task": old_task, "new_task": new_task,
                "protection_source": prot_src,
                "a2_state_in_old_ckpt": bool(
                    a2 and a2.get("protected_slots")),
                "n_protected_slots": len(prot_cols),
                "n_unprotected_slots": len(unprot_cols),
                "n_prot_samples": int(sample_prot.sum()),
                "n_unprot_samples": int((~sample_prot).sum()),
                "key_row_norm_old_mean": float(kn_old.mean()),
                "key_row_norm_old_p10": float(
                    torch.quantile(kn_old, 0.10)),
                "key_row_norm_old_min": kn_old_min,
                "key_row_norm_new_mean": float(kn_new.mean()),
                "key_row_norm_new_p10": float(
                    torch.quantile(kn_new, 0.10)),
                "key_row_norm_new_min": kn_new_min,
                "key_row_norm_new_prot_mean": kn_new_prot_mean,
                "key_row_norm_new_unprot_mean": kn_new_unprot_mean,
                "n_key_rows_below_1e-6": n_below_floor,
                "task_loss": task_loss,
                "base_grad_norm": base_grad_norm,
                "pull_grad_norm": pull_grad_norm,
                "protected_flip_rate": prot_flip,
                "unprotected_flip_rate": unprot_flip,
                "flip_rate_small_margin": flip_small,
                "flip_rate_large_margin": flip_large,
                "old_margin_mean_prot": m_mean,
                "old_margin_median_prot": m_med,
                "old_margin_p10_prot": m_p10,
            }
            print(f"client {cid}: old_task={old_task} new_task={new_task} "
                  f"protected={len(prot_cols)}/{pool_size} "
                  f"task_loss={task_loss:.4f} "
                  f"flip prot={prot_flip:.3f} (n={int(sample_prot.sum())}) "
                  f"unprot={unprot_flip:.3f} (n={int((~sample_prot).sum())}) "
                  f"(small-m {flip_small:.3f} / large-m {flip_large:.3f}) "
                  f"base||gK||={base_grad_norm:.2e} "
                  f"keyRowNorm old={float(kn_old.mean()):.3f}/"
                  f"new={float(kn_new.mean()):.3f}",
                  flush=True)
    finally:
        server.model.train(vit_was_training)
        server.origin_model.train(orig_was_training)
        restore_rng_state(rng_snapshot)

    if not rows:
        raise RuntimeError("no client produced rows; nothing to report")

    # ---------------- persist CSV ----------------
    out_csv = Path(args.out_csv) if args.out_csv else (
        Path(args.run_dir) / "metrics" / "a3_calibration.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out_csv} ({len(rows)} rows)")

    # ---------------- aggregate over clients per tau ----------------
    def col(tau, name):
        return [float(r[name]) for r in rows if r["temp"] == tau]

    def nmean(tau, name):
        vals = col(tau, name)
        return float(np.nanmean(vals)) if vals else float('nan')

    task_losses = [c["task_loss"] for c in per_client.values()]
    task_loss_mean = float(np.mean(task_losses))
    base_grad_mean = float(np.mean(
        [c["base_grad_norm"] for c in per_client.values()]))
    tau_sweep = {}
    for tau in temps:
        psk = nmean(tau, "protected_sample_kl")
        kl_g = nmean(tau, "kl_grad_norm_prot")
        kl_g_old = nmean(tau, "kl_grad_norm_prot_featold")
        lam_grad = (base_grad_mean / kl_g
                    if np.isfinite(kl_g) and kl_g > 0 else float('nan'))
        lam_grad_old = (base_grad_mean / kl_g_old
                        if np.isfinite(kl_g_old) and kl_g_old > 0
                        else float('nan'))
        tau_sweep[tau] = {
            "top1_prob_mean_old": nmean(tau, "top1_prob_mean_old"),
            "top1_prob_old_prot": nmean(tau, "top1_prob_old_prot"),
            "top1_prob_mean_new": nmean(tau, "top1_prob_mean_new"),
            "top1_prob_mean": nmean(tau, "top1_prob_mean"),
            "entropy_mean_old": nmean(tau, "entropy_mean_old"),
            "entropy_mean_new": nmean(tau, "entropy_mean_new"),
            "entropy_mean": nmean(tau, "entropy_mean"),
            "protected_sample_kl_mean": psk,
            "unprotected_sample_kl_mean": nmean(tau, "unprotected_sample_kl"),
            "within_protected_kl_mean": nmean(tau, "within_protected_kl"),
            "full_kl_mean": nmean(tau, "full_kl"),
            "prompt_only_kl_mean": nmean(tau, "prompt_only_kl"),
            "prompt_only_kl_prot_mean": nmean(tau, "prompt_only_kl_prot"),
            "key_only_kl_mean": nmean(tau, "key_only_kl"),
            "key_only_kl_prot_mean": nmean(tau, "key_only_kl_prot"),
            "kl_grad_norm_prot_mean": kl_g,
            "kl_grad_norm_prot_featold_mean": kl_g_old,
            "kl_grad_norm_all_mean": nmean(tau, "kl_grad_norm_all"),
            "lambda_grad": lam_grad,
            "lambda_grad_featold": lam_grad_old,
            "lambda_loss": nmean(tau, "lambda_loss"),
            "kl_rowgrad_top10_share": nmean(tau, "kl_rowgrad_top10_share"),
            "kl_rowgrad_prot_rows_share": nmean(
                tau, "kl_rowgrad_prot_rows_share"),
            "kl_rowgrad_top10_sq_share": nmean(
                tau, "kl_rowgrad_top10_sq_share"),
            "kl_rowgrad_prot_rows_sq_share": nmean(
                tau, "kl_rowgrad_prot_rows_sq_share"),
        }

    # ---------------- verdicts ----------------
    tau_window = [t for t in temps
                  if TOP1_WINDOW[0] <= tau_sweep[t]["top1_prob_mean"]
                  <= TOP1_WINDOW[1]]
    pool_taus = tau_window if tau_window else temps
    rec_tau = min(pool_taus, key=lambda t: abs(
        tau_sweep[t]["top1_prob_mean"] - TOP1_TARGET))
    rec = tau_sweep[rec_tau]

    p_flip = float(np.nanmean(
        [c["protected_flip_rate"] for c in per_client.values()]))
    u_flip = float(np.nanmean(
        [c["unprotected_flip_rate"] for c in per_client.values()]))
    n_prot_samp = [c["n_prot_samples"] for c in per_client.values()]
    n_unprot_samp = [c["n_unprot_samples"] for c in per_client.values()]
    kn_old_mean = float(np.mean(
        [c["key_row_norm_old_mean"] for c in per_client.values()]))
    kn_old_p10 = float(np.mean(
        [c["key_row_norm_old_p10"] for c in per_client.values()]))
    kn_new_mean = float(np.mean(
        [c["key_row_norm_new_mean"] for c in per_client.values()]))
    flip_small = float(np.nanmean(
        [c["flip_rate_small_margin"] for c in per_client.values()]))
    flip_large = float(np.nanmean(
        [c["flip_rate_large_margin"] for c in per_client.values()]))
    p_js = nmean(rec_tau, "protected_js")
    u_js = nmean(rec_tau, "unprotected_js")
    flip_ratio = (p_flip / u_flip
                  if np.isfinite(p_flip) and np.isfinite(u_flip) and u_flip > 0
                  else float('nan'))
    separation_holds = bool(np.isfinite(p_flip) and np.isfinite(u_flip)
                            and p_flip > u_flip)
    if separation_holds:
        sep_verdict = (f"SEPARATION PRESENT: protected flip {p_flip:.3f} > "
                       f"unprotected {u_flip:.3f} (ratio "
                       f"{flip_ratio:.2f}x)")
    elif np.isfinite(p_flip) and np.isfinite(u_flip):
        sep_verdict = (f"NO SEPARATION (informational, not a negation of "
                       f"the mask): protected flip {p_flip:.3f} <= "
                       f"unprotected {u_flip:.3f} (ratio "
                       f"{flip_ratio:.2f}x). The protection set means "
                       f"'history tasks NEED these slots' — it does NOT "
                       f"require protected slots to flip more than "
                       f"unprotected ones; unprotected flips are expected "
                       f"(new-task learning rewrites free slots). Validate "
                       f"the mask IN TRAINING by whether the constraint "
                       f"improves OLD-task accuracy. Coverage: protected "
                       f"samples per client {n_prot_samp}, unprotected "
                       f"{n_unprot_samp} — small unprotected groups make "
                       f"the unprotected rate noisy")
    else:
        sep_verdict = "INDETERMINATE: empty protected or unprotected group"

    # sharpness evidence: PROTECTED-sample old top-1 prob when available
    # (the verdict is about the samples A3 constrains), all-sample mean
    # as fallback
    old_top1_for_verdict = (rec["top1_prob_old_prot"]
                            if np.isfinite(rec.get("top1_prob_old_prot",
                                                   float('nan')))
                            else rec["top1_prob_mean_old"])
    kl_verdict, kl_effective = make_kl_verdict(
        psk=rec["protected_sample_kl_mean"],
        old_top1=old_top1_for_verdict,
        kl_grad=rec["kl_grad_norm_prot_mean"],
        base_grad=base_grad_mean,
        lam_grad=rec["lambda_grad"],
        prot_flip=p_flip)

    notes = [
        "Tail_Anchor's key/anchor pool (200 rows) is a ROUTING pool, not a "
        "class-indexed prototype list: the official forward routes over "
        "ALL 200 slots regardless of nb_classes, any slot can receive mass "
        "and be optimized, so the diagnostic keeps the full pool",
        "routing similarity is taken BEFORE anchor selection; the anchor "
        "component of the interface never enters the routing math",
        "flip rates are conditioned on the OLD interface's top-1 slot "
        "belonging to the group; argmax is tau-invariant so flip rates are "
        "constant across tau rows",
        "per-column JS: slot j's profile = R[:, j] renormalized over "
        "reference samples; JS = sqrt(JSD) between the two interfaces",
        "reference inputs are public surrogate data (kd_fusion_prompt "
        "protocol), deterministic per-client sample; identical batch "
        "composition in both interface passes (batchwise prompt voting "
        "cancels)",
        "protection set is ALWAYS replicated at the A3 snapshot moment "
        "(tasks 0..old_task top-1 under the old key, ViT task_id=old_task+1, "
        "batch=16 hardcoded as in the official A2); the a2_state stored in "
        "old_ckpt is one task stale and is recorded but never used",
        "MECHANISTIC NOTE: in the official phase-2 loss the key's ONLY "
        "continuous gradient channel is -0.1*pull + Adam weight decay — CE "
        "and InfoNCE reach the key only through the discrete topk index. "
        "lambda_grad (base/key-gradient norm vs KL/key-gradient norm) is "
        "therefore the primary lambda estimate; lambda_loss (CE/KL) is a "
        "coarse reference only",
        "lambda decisions should use ADJACENT checkpoint pairs (one task of "
        "drift, e.g. task_03_end -> task_04_end): A3 snapshots R_old at "
        "every task start. Multi-task spans remain useful as "
        "accumulated-drift diagnosis",
    ]
    notes.append(
        "lambda_grad balances gradient norms at one point (the new_ckpt "
        "interface); Adam normalizes per-parameter update magnitudes, so "
        "treat lambda_grad as the CENTER of the lambda sweep grid (scan at "
        "least +-1 order of magnitude), never as an exact training-scale "
        "equivalence")
    if a2_run_drift_suppressed:
        notes.append(
            f"this is a key-frozen A2 run (arm={a2_arm}): protected key rows "
            f"are restored after every optimizer step, so routing drift on "
            f"protected slots (protected-sample KL, protected flip rate) is "
            f"SUPPRESSED relative to a pure fedta baseline — all drift "
            f"numbers are UNDERESTIMATED here and the recommended lambda "
            f"is NOT deployment-valid; re-run on a pure fedta run before "
            f"finalizing lambda and the loss form")

    # key-norm detail aggregates (used by both the JSON dump and console)
    kn_min_new = float(np.nanmean(
        [c["key_row_norm_new_min"] for c in per_client.values()]))
    kn_min_old = float(np.nanmean(
        [c["key_row_norm_old_min"] for c in per_client.values()]))
    kn_prot_m = float(np.nanmean(
        [c["key_row_norm_new_prot_mean"] for c in per_client.values()]))
    kn_unprot_m = float(np.nanmean(
        [c["key_row_norm_new_unprot_mean"] for c in per_client.values()]))
    n_floor = int(sum(c["n_key_rows_below_1e-6"]
                      for c in per_client.values()))

    # ---------------- persist JSON ----------------
    out_json = out_csv.with_suffix(".json")
    meta = {
        "script": "diagnostics/a3_offline_calibration.py",
        "run_dir": str(Path(args.run_dir)),
        "method": method, "a2_arm": a2_arm,
        "a2_run_drift_suppressed": a2_run_drift_suppressed,
        "old_ckpt": old_path, "new_ckpt": new_path,
        "old_task": sorted({c["old_task"] for c in per_client.values()}),
        "new_task": sorted({c["new_task"] for c in per_client.values()}),
        "seed": seed, "n_ref": n_ref,
        "ref_batch_size": args.ref_batch_size, "temps": temps,
        "device": args.device,
        "surrogate_pool_size": n_pool,
        "clients": {str(k): v for k, v in per_client.items()},
        "task_loss_mean": task_loss_mean,
        "base_grad_norm_mean": base_grad_mean,
        "tau_sweep": {str(t): tau_sweep[t] for t in temps},
        "tau_window_top1_0.6_0.9": tau_window,
        "recommended_tau": rec_tau,
        "verdicts": {
            "tau": {
                "window": tau_window, "recommended": rec_tau,
                "recommended_top1_prob": rec["top1_prob_mean"],
                "criterion": "client-mean top-1 prob of softmax(tau*sim), "
                             "pooled over both interfaces, in [0.6, 0.9]; "
                             "recommended = closest to 0.75",
            },
            "kl_lambda": {
                "protected_sample_kl_mean": rec["protected_sample_kl_mean"],
                "within_protected_kl_mean": rec["within_protected_kl_mean"],
                "full_kl_mean": rec["full_kl_mean"],
                "task_loss_mean": task_loss_mean,
                "base_grad_norm_mean": base_grad_mean,
                "pull_grad_norm_mean": float(np.mean(
                    [c["pull_grad_norm"] for c in per_client.values()])),
                "kl_grad_norm_prot_mean": rec["kl_grad_norm_prot_mean"],
                "kl_grad_norm_prot_featold_mean":
                    rec["kl_grad_norm_prot_featold_mean"],
                "prompt_only_kl_prot_mean":
                    rec["prompt_only_kl_prot_mean"],
                "key_only_kl_prot_mean": rec["key_only_kl_prot_mean"],
                "old_top1_prob_mean": rec["top1_prob_mean_old"],
                "lambda_loss": rec["lambda_loss"],
                "lambda_grad": rec["lambda_grad"],
                "lambda_grad_featold": rec["lambda_grad_featold"],
                "kl_rowgrad_top10_share": rec["kl_rowgrad_top10_share"],
                "kl_rowgrad_prot_rows_share":
                    rec["kl_rowgrad_prot_rows_share"],
                "kl_rowgrad_top10_sq_share":
                    rec["kl_rowgrad_top10_sq_share"],
                "kl_rowgrad_prot_rows_sq_share":
                    rec["kl_rowgrad_prot_rows_sq_share"],
                "old_top1_prob_prot": rec["top1_prob_old_prot"],
                "key_row_norm_old_mean": kn_old_mean,
                "key_row_norm_old_p10": kn_old_p10,
                "key_row_norm_old_min_mean": kn_min_old,
                "key_row_norm_new_mean": kn_new_mean,
                "key_row_norm_new_min_mean": kn_min_new,
                "key_row_norm_new_prot_mean": kn_prot_m,
                "key_row_norm_new_unprot_mean": kn_unprot_m,
                "n_key_rows_below_1e-6_total": n_floor,
                "top_grad_rows_per_client": {
                    str(cid): [
                        {"slot": int(s),
                         "protected": bool(
                             rowgrad_detail[(cid, rec_tau)]["prot_mask"][s]),
                         "key_norm_new": float(
                             rowgrad_detail[(cid, rec_tau)]["kn_new"][s]),
                         "row_grad_l2": float(
                             rowgrad_detail[(cid, rec_tau)]["row_g"][s])}
                        for s in rowgrad_detail[(cid, rec_tau)]["row_g"]
                        .topk(min(10, rowgrad_detail[(cid, rec_tau)]
                                  ["row_g"].numel())).indices.tolist()]
                    for cid in sorted({c for (c, t) in rowgrad_detail
                                       if t == rec_tau})},
                "deployment_valid": not a2_run_drift_suppressed,
                "effective": kl_effective,
                "verdict": kl_verdict,
            },
            "mask_separation": {
                "protected_flip_rate_mean": p_flip,
                "unprotected_flip_rate_mean": u_flip,
                "flip_ratio": flip_ratio,
                "n_prot_samples_per_client": n_prot_samp,
                "n_unprot_samples_per_client": n_unprot_samp,
                "flip_rate_small_margin_mean": flip_small,
                "flip_rate_large_margin_mean": flip_large,
                "old_margin_mean_prot": float(np.nanmean(
                    [c["old_margin_mean_prot"]
                     for c in per_client.values()])),
                "old_margin_p10_prot": float(np.nanmean(
                    [c["old_margin_p10_prot"]
                     for c in per_client.values()])),
                "protected_js_mean": p_js,
                "unprotected_js_mean": u_js,
                "holds": separation_holds,
                "verdict": sep_verdict,
            },
        },
        "notes": notes,
        "n_clients_processed": len(per_client),
        "n_clients_skipped": n_skipped,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"saved {out_json}")

    # ---------------- console summary (three verdicts) ----------------
    print(f"\n===== T-a: tau sweep (client-mean, pooled over interfaces) =====")
    for tau in temps:
        s = tau_sweep[tau]
        print(f"tau={tau:g}: top1={s['top1_prob_mean']:.4f} "
              f"(old {s['top1_prob_mean_old']:.4f} / "
              f"new {s['top1_prob_mean_new']:.4f})  "
              f"entropy={s['entropy_mean']:.3f}  "
              f"prot_sample_kl={s['protected_sample_kl_mean']:.4f}  "
              f"kl||gradK||={s['kl_grad_norm_prot_mean']:.2e}")
    print(f"tau window with top1 in [{TOP1_WINDOW[0]}, {TOP1_WINDOW[1]}]: "
          f"{tau_window}  -> recommended tau={rec_tau:g} "
          f"(top1={rec['top1_prob_mean']:.4f})")

    print(f"\n===== T-b: KL flavors, gradients & lambda (at tau={rec_tau:g}) "
          f"=====")
    if a2_run_drift_suppressed:
        print("[A2-RUN: key frozen — drift numbers suppressed, recommended "
              "lambda NOT deployment-valid; rerun on a pure fedta baseline]")
    print(f"protected-sample full KL (PRIMARY) mean="
          f"{rec['protected_sample_kl_mean']:.4f}  per client: "
          f"{['%.4f' % v for v in col(rec_tau, 'protected_sample_kl')]}")
    print(f"within-mask renormalized KL (auxiliary) mean="
          f"{rec['within_protected_kl_mean']:.4f}")
    print(f"full-pool KL mean={rec['full_kl_mean']:.4f}")
    print(f"drift decomposition, protected samples (single-factor terms, "
          f"non-additive — do not sum them or convert to attribution "
          f"shares): key-only KL={rec['key_only_kl_prot_mean']:.4f}  "
          f"prompt-only KL={rec['prompt_only_kl_prot_mean']:.4f}  "
          f"total={rec['protected_sample_kl_mean']:.4f}")
    print(f"task CE mean={task_loss_mean:.4f} -> lambda_loss "
          f"(coarse, NOT a training lambda) ~ {rec['lambda_loss']:.2e}")
    print(f"base key-gradient ||-0.1*grad pull + wd*K|| mean="
          f"{base_grad_mean:.3e}  vs  ||grad_K KL_prot|| mean="
          f"{rec['kl_grad_norm_prot_mean']:.3e}")
    print(f"lambda_grad (PRIMARY, deployment feat_new) ~ "
          f"{rec['lambda_grad']:.2e}  [sweep center, scan +-1 order]"
          if np.isfinite(rec['lambda_grad'])
          else "lambda_grad: undefined (KL key-gradient ~ 0)")
    print(f"KL gradient isolation variant (feat_old, prompts undrifted): "
          f"||grad_K KL_prot||={rec['kl_grad_norm_prot_featold_mean']:.3e} "
          f"-> lambda_grad_featold ~ "
          f"{rec['lambda_grad_featold']:.2e}"
          if np.isfinite(rec['lambda_grad_featold'])
          else "KL isolation gradient: undefined")
    if (np.isfinite(rec['lambda_grad'])
            and np.isfinite(rec['lambda_grad_featold'])
            and rec['lambda_grad_featold'] > 0):
        lam_ratio = rec['lambda_grad'] / rec['lambda_grad_featold']
        lever = ('weaker' if lam_ratio > 1.1
                 else 'stronger' if lam_ratio < 0.9 else 'about equal')
        print(f"lambda ratio (deployment/isolation) {lam_ratio:.2f}x: the "
              f"deployment constraint's lever on the key is "
              f"{lever} than under "
              f"undrifted prompts — the DEPLOYMENT value is the diagnostic "
              f"reference for the sweep center. Factor attribution is "
              f"carried by the drift decomposition above, not by this "
              f"ratio")
    print(f"gradient concentration (protected-sample KL, this tau): "
          f"top-10 key rows carry {rec['kl_rowgrad_top10_share']:.1%} of "
          f"row-grad mass (L1) / {rec['kl_rowgrad_top10_sq_share']:.1%} of "
          f"squared norm (L2, aligned with lambda's denominator); "
          f"protected rows carry {rec['kl_rowgrad_prot_rows_share']:.1%} "
          f"(L1) / {rec['kl_rowgrad_prot_rows_sq_share']:.1%} (L2)")
    print(f"key row norms (parameter scale): old mean={kn_old_mean:.4f} "
          f"min={kn_min_old:.2e}  new mean={kn_new_mean:.4f} "
          f"min={kn_min_new:.2e}  new protected-row mean={kn_prot_m:.4f} "
          f"vs unprotected-row mean={kn_unprot_m:.4f}  "
          f"rows below 1e-6 (old+new, all clients): {n_floor}")
    print("top gradient rows (per client, top-5 by row-grad L2):")
    for cid in sorted({c for (c, t) in rowgrad_detail if t == rec_tau}):
        d = rowgrad_detail[(cid, rec_tau)]
        top_idx = d["row_g"].topk(min(5, d["row_g"].numel())).indices
        cells = [f"slot{s}({'P' if d['prot_mask'][s] else 'u'}) "
                 f"kn={d['kn_new'][s]:.2e} g={d['row_g'][s]:.2e}"
                 for s in top_idx.tolist()]
        print(f"  client {cid}: " + " | ".join(cells))
    print("note: gradient concentration alone does NOT prove a "
          "parameter-scale artifact — real routing change also "
          "concentrates; check the top rows' key norms (tiny kn with "
          "huge g = artifact; healthy kn with huge g = real drift)")
    print(f"verdict: {kl_verdict}")

    print(f"\n===== T-c: protection-mask separation (at tau={rec_tau:g}) =====")
    print(f"top-1 flip rate: protected={p_flip:.3f} (n/client {n_prot_samp})  "
          f"unprotected={u_flip:.3f} (n/client {n_unprot_samp})  "
          f"ratio={flip_ratio:.2f}x")
    print(f"flip stratification by old margin (protected samples): "
          f"small-margin={flip_small:.3f}  large-margin={flip_large:.3f}  "
          f"(margin mean={float(np.nanmean([c['old_margin_mean_prot'] for c in per_client.values()])):.4f}, "
          f"p10={float(np.nanmean([c['old_margin_p10_prot'] for c in per_client.values()])):.4f})")
    print(f"per-column JS distance: protected={p_js:.4f}  "
          f"unprotected={u_js:.4f}")
    print(f"verdict: {sep_verdict}")
    for note in notes:
        if note.startswith(("this is a key-frozen", "MECHANISTIC")):
            print(f"note: {note}")


if __name__ == "__main__":
    main()
