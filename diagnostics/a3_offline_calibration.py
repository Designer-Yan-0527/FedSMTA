"""A3 offline calibration: routing-consistency loss design parameters.

Pre-training calibration for method A3 (functional-level routing consistency
loss  L = L_task + lambda * KL(R_old || R_new),  R = softmax(tau * sim(z, K))).
Pure inference, zero training. Answers the three unknowns before any A3
training code is written:

  T-a  temperature tau:
       Forward the SAME reference inputs (public surrogate data, deterministic
       per-client sample of --n_ref items, identical batch composition in both
       passes) through both generation interfaces, take the raw key-routing
       similarity matrices (B x pool_size, computed exactly like
       Tail_Anchor.forward: l2-normalized features . l2-normalized key rows,
       taken BEFORE anchor selection), sweep tau over --temps and report the
       top-1 probability and entropy (nats) of R = softmax(tau * sim).
       Target: the tau window where top-1 probability is ~[0.6, 0.9].

  T-b  KL magnitude (the decisive number):
       KL(R_old || R_new) per client on the reference inputs, both masked to
       the protection-set slots (renormalized within the mask) and unmasked
       (full pool), PLUS a task-loss magnitude reference: one batch of the
       current (new_ckpt) task's train data forwarded normally under the new
       interface with the official class mask, cross-entropy mean. The
       lambda ~ L_task / KL magnitude then reads off directly.
       Verdict: if the two-generation routing KL is naturally < 0.01, the
       damage concentrates in hard top-1 flips rather than distribution
       drift, a KL constraint would be ineffective and A3 should switch to
       a margin-hinge form. This conclusion is printed and stored.

  T-c  protection-set mask separation:
       top-1 flip rate (old-vs-new argmax disagreement, conditioned on the
       OLD interface's top-1 destination being a protected slot) and mean
       per-column JS distance (column j: the distribution over reference
       samples routed to slot j, old vs new; JS = sqrt(JSD)), computed
       separately for protected and unprotected slots. Verifies the mask
       design premise "drift concentrates where the constraint is needed".

Interface definitions (aligned with compatibility_drift.py's 4-combo):
  old = (old prompt, old key, old anchor)          from --old_ckpt
  new = (current prompt, current key, current anchor) from --new_ckpt
Routing similarity depends only on the key and the features (the anchor is
selected AFTER routing), so anchors never enter the routing math; the new
anchor/head only matter for the task-loss forward.

Protection set: ALWAYS replicated from the A2 mechanism at the A3 snapshot
moment (= the start of the task FOLLOWING old_ckpt): top-1 routing of ALL
completed tasks' (0..old_task INCLUSIVE) train-split features under the OLD
key, union of hit slots (sequential loader, shuffle=False, num_workers=0,
eval mode — RNG-neutral, mirrors Client_DF._a2_init_task_protection with
self.task_id = old_task + 1, whose `completed` is exactly range(old_task+1)
and whose "current key at task start" is exactly the old_ckpt key).
The a2_state stored in old_ckpt is deliberately NOT used: it was computed at
the START of old_ckpt's own task (tasks 0..old_task-1, older key) — one task
stale relative to the A3 snapshot moment. Its presence is only recorded.

RNG discipline (same as compatibility_drift.py): the whole diagnostic runs
under an RNG snapshot that is restored at the end; both interface passes are
seeded with the fixed per-(client, task) convention 9700 + 100*cid + task;
per-client reference/batch sampling uses detached torch.Generator objects so
the global stream is never consumed; the current vit prompts are reloaded
into the shared vit after every old-interface pass.

Outputs:
  CSV  (default <run_dir>/metrics/a3_calibration.csv): one row per
       (client x tau): top1/entropy stats (per interface + pooled),
       masked_kl, full_kl, protected/unprotected flip rates and JS.
  JSON (same directory, .csv suffix -> .json): metadata, tau sweep summary,
       task-loss reference, and the three verdicts.
  Console: per-client lines + the three-part verdict summary.

Usage (repo root, on the server):
  python diagnostics/a3_offline_calibration.py \
      --run_dir output/cifar100/fedta_a2/a2_key/seed_42 \
      --old_ckpt checkpoints/task_02_end.pth \
      --new_ckpt checkpoints/task_04_end.pth \
      --n_ref 64
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

# KL below this magnitude is judged "naturally negligible" for A3 (T-b)
KL_EFFECTIVE_FLOOR = 0.01

# T-a target window for softmax top-1 probability
TOP1_WINDOW = (0.6, 0.9)
TOP1_TARGET = 0.75


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
    p.add_argument("--batch_size", type=int, default=16)
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


def interface_features(client, prompt_module, ref_x, task_id, batch_size):
    """Features of the fixed reference tensor under one interface.
    The shared vit gets the interface's prompts loaded (caller restores the
    current ones afterwards). Eval mode is assumed (set once in main).
    Mirrors evaluate()/_a2_init_task_protection: vit forward with the
    checkpoint-time convention task_id=<current task>, train=True
    (use_prompt_mask=False in this codebase, so the flag is inert)."""
    client.vit.load_prompts(prompt_module)
    feats = []
    for i in range(0, ref_x.shape[0], batch_size):
        x = ref_x[i:i + batch_size].to(client.device, non_blocking=True)
        with torch.no_grad():
            output = client.original_model(x)
            cls_features = output['pre_logits']
            output = client.vit(x, task_id=task_id,
                                cls_features=cls_features, train=True)
        feats.append(output['feat'].to(client.device))
    return torch.cat(feats, dim=0)


def get_train_split(client, task_k):
    """data_split_indices keys may be int (fresh) or str (post-serialize)."""
    dsi = client.data_split_indices
    if task_k in dsi:
        return dsi[task_k]["train"]
    if str(task_k) in dsi:
        return dsi[str(task_k)]["train"]
    raise KeyError(f"data_split_indices missing task {task_k}")


def replicate_protection(client, old_prompt_module, old_key, old_task,
                         batch_size):
    """A2 protection-set replication at the old-checkpoint time point:
    union over completed tasks 0..old_task of the top-1 routing destinations
    of their train-split features under the OLD key (i.e. exactly what
    Client_DF._a2_init_task_protection computes when the task FOLLOWING
    old_ckpt starts, which is the moment A3 would snapshot R_old).
    RNG-neutral like the A2 mechanism: sequential loader, shuffle=False,
    num_workers=0, eval mode."""
    key = old_key.to(client.device)
    client.vit.load_prompts(old_prompt_module)
    hit_slots = []
    for t in range(0, old_task + 1):
        dataset = Subset(client.train_data[t], get_train_split(client, t))
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                            shuffle=False)
        feats = []
        for input, _target in loader:
            input = input.to(client.device, non_blocking=True)
            with torch.no_grad():
                output = client.original_model(input)
                cls_features = output['pre_logits']
                output = client.vit(input, task_id=old_task,
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


def task_loss_reference(client, gen, batch_size, nb_classes):
    """One deterministic batch of the current task's train data, forwarded
    normally under the NEW interface; official train() cross-entropy
    (Tail_Anchor anchor-top-1 + current head, classes outside current_class
    masked to -inf over the first nb_classes logits — exact train() replica;
    the pull/InfoNCE terms are excluded on purpose: only the CE magnitude
    reference is wanted)."""
    n = len(client.traindata)
    take = min(batch_size, n)
    idx = torch.randperm(n, generator=gen)[:take].tolist()
    xs, ys = [], []
    for i in idx:
        x, y = client.traindata[i]
        xs.append(x)
        ys.append(int(y))
    x = torch.stack(xs).to(client.device, non_blocking=True)
    target = torch.tensor(ys, dtype=torch.long).to(client.device)

    with torch.no_grad():
        output = client.original_model(x)
        cls_features = output['pre_logits']
        output = client.vit(x, task_id=client.task_id,
                            cls_features=cls_features, train=True)
        feat = output['feat'].to(client.device)
        # Tail_Anchor.forward replica (without its hardcoded .to('cuda'))
        similarity = routing_similarity(feat, client.model.key.detach())
        _, index = torch.topk(similarity, k=1)
        anchor = client.model.anchor_pool[index].reshape(-1, feat.shape[1])
        x1 = torch.stack((feat, anchor), dim=1).reshape(-1, feat.shape[1] * 2)
        logits = client.model.head(x1)
        # official class mask (train(): only the first nb_classes logits are
        # masked; head outputs beyond nb_classes are part of official CE)
        not_mask = np.setdiff1d(np.arange(nb_classes), client.current_class)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(client.device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        return float(F.cross_entropy(logits, target))


def masked_kl(log_rold, log_rnew, cols):
    """Per-sample KL(R_old || R_new) restricted to the protection-set columns,
    both distributions renormalized within the mask. Returns scalar mean."""
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

            # ---- protection set: ALWAYS replicate at the A3 snapshot moment.
            # a2_state in old_ckpt is one task STALE (computed at the start
            # of old_ckpt's own task); only its presence is recorded.
            a2 = old_state.get("a2_state")
            torch.manual_seed(SEED_BASE + 100 * cid + old_task)
            slots = replicate_protection(client, old_prompt, old_key,
                                          old_task, args.batch_size)
            prot_cols = sorted(int(s) for s in slots.tolist())
            prot_src = (f"replicated A2 mechanism (old key over train "
                        f"splits of tasks 0..{old_task})")
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
                                          args.batch_size)
            sim_old = routing_similarity(feat_old, old_key.to(client.device))

            # ---- new interface pass (current prompts restored first) ----
            torch.manual_seed(SEED_BASE + 100 * cid + new_task)
            feat_new = interface_features(client, cur_prompt, ref_x, new_task,
                                          args.batch_size)
            sim_new = routing_similarity(feat_new, cur_key.to(client.device))

            # ---- task-loss magnitude reference (new interface) ----
            gen2 = torch.Generator()
            gen2.manual_seed(SEED_BASE + 100 * cid + new_task + 1)
            task_loss = task_loss_reference(client, gen2, args.batch_size,
                                            nb_classes)

            # ---- T-c flip rates (tau-independent: argmax is tau-invariant) -
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

            # ---- T-a / T-b per tau ----
            for tau in temps:
                log_ro = F.log_softmax(tau * sim_old, dim=1)
                log_rn = F.log_softmax(tau * sim_new, dim=1)
                ro, rn = log_ro.exp(), log_rn.exp()
                top1_o = float(ro.max(dim=1).values.mean())
                top1_n = float(rn.max(dim=1).values.mean())
                h_o = float((-(ro * log_ro).sum(dim=1)).mean())
                h_n = float((-(rn * log_rn).sum(dim=1)).mean())
                full_kl = float(((ro * (log_ro - log_rn)).sum(dim=1)).mean())
                if prot_cols:
                    m_kl = masked_kl(log_ro, log_rn, prot_cols)
                    js_cols = column_js_distance(ro, rn)
                    p_js = float(js_cols[prot_cols].mean())
                else:
                    m_kl, p_js = float('nan'), float('nan')
                u_js = (float(column_js_distance(ro, rn)[unprot_cols].mean())
                        if unprot_cols else float('nan'))
                rows.append({
                    "client_id": cid,
                    "old_task": old_task,
                    "new_task": new_task,
                    "temp": tau,
                    "top1_prob_mean_old": f"{top1_o:.6f}",
                    "top1_prob_mean_new": f"{top1_n:.6f}",
                    "top1_prob_mean": f"{(top1_o + top1_n) / 2:.6f}",
                    "entropy_mean_old": f"{h_o:.6f}",
                    "entropy_mean_new": f"{h_n:.6f}",
                    "entropy_mean": f"{(h_o + h_n) / 2:.6f}",
                    "masked_kl": f"{m_kl:.6f}",
                    "full_kl": f"{full_kl:.6f}",
                    "protected_flip_rate": f"{prot_flip:.4f}",
                    "unprotected_flip_rate": f"{unprot_flip:.4f}",
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
                "task_loss": task_loss,
                "protected_flip_rate": prot_flip,
                "unprotected_flip_rate": unprot_flip,
            }
            print(f"client {cid}: old_task={old_task} new_task={new_task} "
                  f"protected={len(prot_cols)}/{pool_size} [{prot_src}] "
                  f"task_loss={task_loss:.4f} "
                  f"flip prot={prot_flip:.3f} unprot={unprot_flip:.3f}",
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
    tau_sweep = {}
    for tau in temps:
        m_kl = nmean(tau, "masked_kl")
        lam = (task_loss_mean / m_kl
               if np.isfinite(m_kl) and m_kl > 0 else float('nan'))
        tau_sweep[tau] = {
            "top1_prob_mean_old": nmean(tau, "top1_prob_mean_old"),
            "top1_prob_mean_new": nmean(tau, "top1_prob_mean_new"),
            "top1_prob_mean": nmean(tau, "top1_prob_mean"),
            "entropy_mean_old": nmean(tau, "entropy_mean_old"),
            "entropy_mean_new": nmean(tau, "entropy_mean_new"),
            "entropy_mean": nmean(tau, "entropy_mean"),
            "masked_kl_mean": m_kl,
            "full_kl_mean": nmean(tau, "full_kl"),
            "lambda_magnitude": lam,
        }

    # ---------------- verdicts ----------------
    tau_window = [t for t in temps
                  if TOP1_WINDOW[0] <= tau_sweep[t]["top1_prob_mean"]
                  <= TOP1_WINDOW[1]]
    pool_taus = tau_window if tau_window else temps
    rec_tau = min(pool_taus, key=lambda t: abs(
        tau_sweep[t]["top1_prob_mean"] - TOP1_TARGET))
    rec = tau_sweep[rec_tau]
    lam_rec = rec["lambda_magnitude"]

    if np.isfinite(rec["masked_kl_mean"]) and \
            rec["masked_kl_mean"] < KL_EFFECTIVE_FLOOR:
        kl_verdict = (
            f"masked KL = {rec['masked_kl_mean']:.4f} < {KL_EFFECTIVE_FLOOR}: "
            f"two-generation routing distributions are naturally near-"
            f"identical on protected slots; damage concentrates in hard "
            f"top-1 flips, NOT distribution drift -> the KL constraint "
            f"would be INEFFECTIVE; A3 should use a margin-hinge form "
            f"(penalize |tau*(sim_old - sim_new)| on protected slots) "
            f"instead of softmax-KL")
        kl_effective = False
    else:
        kl_verdict = (
            f"masked KL = {rec['masked_kl_mean']:.4f} >= "
            f"{KL_EFFECTIVE_FLOOR}: measurable distribution drift on "
            f"protected slots -> the KL constraint has signal to act on")
        kl_effective = True

    p_flip = float(np.nanmean(
        [c["protected_flip_rate"] for c in per_client.values()]))
    u_flip = float(np.nanmean(
        [c["unprotected_flip_rate"] for c in per_client.values()]))
    p_js = nmean(rec_tau, "protected_js")
    u_js = nmean(rec_tau, "unprotected_js")
    flip_ratio = (p_flip / u_flip
                  if np.isfinite(p_flip) and np.isfinite(u_flip) and u_flip > 0
                  else float('nan'))
    separation_holds = bool(np.isfinite(p_flip) and np.isfinite(u_flip)
                            and p_flip > u_flip)
    if separation_holds:
        sep_verdict = (f"HOLDS: protected flip {p_flip:.3f} > unprotected "
                       f"{u_flip:.3f} (ratio {flip_ratio:.2f}x) — drift "
                       f"concentrates on the slots the mask constrains")
    elif np.isfinite(p_flip) and np.isfinite(u_flip):
        sep_verdict = (f"DOES NOT HOLD: protected flip {p_flip:.3f} <= "
                       f"unprotected {u_flip:.3f} (ratio "
                       f"{flip_ratio:.2f}x)")
    else:
        sep_verdict = "INDETERMINATE: empty protected or unprotected group"

    method = getattr(run_args, "method", None)
    a2_arm = getattr(run_args, "a2_arm", None)
    notes = [
        f"key pool has {rows[0]['n_protected_slots'] + rows[0]['n_unprotected_slots']} "
        f"slots (Tail_Anchor(10,768,200)) vs nb_classes={nb_classes}; slots "
        f">= {nb_classes} are never-trained noise slots, kept in the pool "
        f"because Tail_Anchor.forward routes over all of them",
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
        "(tasks 0..old_task top-1 under the old key); the a2_state stored "
        "in old_ckpt is one task stale (computed at the START of old_ckpt's "
        "own task) and is recorded but never used",
    ]
    if method == 'fedta_a2' and a2_arm in ('key', 'both'):
        notes.append(
            f"this is a key-frozen A2 run (arm={a2_arm}): protected key rows "
            f"are restored after every optimizer step, so routing drift on "
            f"protected slots (masked KL, protected flip rate) is SUPPRESSED "
            f"relative to a pure fedta baseline — KL is UNDERESTIMATED here; "
            f"re-run on a pure fedta run before finalizing lambda")

    # ---------------- persist JSON ----------------
    out_json = out_csv.with_suffix(".json")
    meta = {
        "script": "diagnostics/a3_offline_calibration.py",
        "run_dir": str(Path(args.run_dir)),
        "method": method, "a2_arm": a2_arm,
        "old_ckpt": old_path, "new_ckpt": new_path,
        "old_task": sorted({c["old_task"] for c in per_client.values()}),
        "new_task": sorted({c["new_task"] for c in per_client.values()}),
        "seed": seed, "n_ref": n_ref, "batch_size": args.batch_size,
        "temps": temps, "device": args.device,
        "surrogate_pool_size": n_pool,
        "clients": {str(k): v for k, v in per_client.items()},
        "task_loss_mean": task_loss_mean,
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
                "masked_kl_mean": rec["masked_kl_mean"],
                "full_kl_mean": rec["full_kl_mean"],
                "task_loss_mean": task_loss_mean,
                "lambda_magnitude": lam_rec,
                "effective": kl_effective,
                "verdict": kl_verdict,
            },
            "mask_separation": {
                "protected_flip_rate_mean": p_flip,
                "unprotected_flip_rate_mean": u_flip,
                "flip_ratio": flip_ratio,
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
              f"masked_kl={s['masked_kl_mean']:.4f}")
    print(f"tau window with top1 in [{TOP1_WINDOW[0]}, {TOP1_WINDOW[1]}]: "
          f"{tau_window}  -> recommended tau={rec_tau:g} "
          f"(top1={rec['top1_prob_mean']:.4f})")

    print(f"\n===== T-b: KL magnitude & lambda (at tau={rec_tau:g}) =====")
    per_client_kl = col(rec_tau, "masked_kl")
    print(f"masked KL(R_old||R_new) mean={rec['masked_kl_mean']:.4f}  "
          f"per client: {['%.4f' % v for v in per_client_kl]}")
    print(f"full   KL mean={rec['full_kl_mean']:.4f}")
    print(f"task-loss CE mean={task_loss_mean:.4f} "
          f"(per client: {['%.4f' % c['task_loss'] for c in per_client.values()]})")
    print(f"lambda ~ L_task/KL = {lam_rec:.2f} "
          f"(order 10^{int(np.log10(lam_rec))})" if np.isfinite(lam_rec) and lam_rec > 0
          else "lambda ~ L_task/KL: undefined (masked KL ~ 0)")
    print(f"verdict: {kl_verdict}")

    print(f"\n===== T-c: protection-mask separation (at tau={rec_tau:g}) =====")
    print(f"top-1 flip rate: protected={p_flip:.3f}  "
          f"unprotected={u_flip:.3f}  ratio={flip_ratio:.2f}x")
    print(f"per-column JS distance: protected={p_js:.4f}  "
          f"unprotected={u_js:.4f}")
    print(f"verdict: {sep_verdict}")
    for note in notes:
        if note.startswith(("this is a key-frozen",)):
            print(f"note: {note}")


if __name__ == "__main__":
    main()
