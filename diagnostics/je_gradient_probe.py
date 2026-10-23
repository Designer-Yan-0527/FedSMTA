"""A2-JE gradient-consistency probe (zero-training diagnostic, read-only).

Mechanism question (pre-registered): at the existing task-end checkpoints,
on the SAME local-training batches under the SAME model state, how much do
the Phase-1 prompt objective and the DEPLOYED classification path disagree
about the update direction of the prompt parameters?

Background (verified against Client_DF.train()):
  - Phase-1 optimizes vit prompts with  CE(vit_internal_head_logits) -
    0.1 * reduce_sim  (the internal head is scaffolding, not deployed).
  - Phase-2 trains Tail_Anchor/Chead with the vit forward inside
    torch.no_grad() -> prompts are NEVER optimized for the deployed Chead
    decision boundary (the A2-JE premise).
  - Deployment: original_model(no_grad) -> vit -> model(Tail_Anchor) ->
    Chead 'pre', task-masked.

Four gradients per batch (P = prompt parameters only; head gradients are
recorded separately and NEVER mixed into the prompt cosines):
  G1 = grad_P CE_phase1          (masked CE on vit internal-head logits)
  G2 = grad_P(-0.1 * reduce_sim) (Phase-1 regularizer, vit-side reduce_sim)
  G3 = grad_P(CE_phase1 - 0.1 * reduce_sim)   (full Phase-1 objective)
  G4 = grad_P CE_deployed        (masked CE on Chead 'pre' via the
       deployment forward; computed WITH grad through vit -- the official
       Phase-2 no_grad is exactly the cut this probe measures THROUGH, so
       a broken/zero G4 is reported as 'gradient blocked', not as
       'direction inconsistent')

Faithfulness: G1-G3 replicate Client_DF.train() phase-1 L306-333 exactly
(original_model under no_grad, vit(task_id, train=True), index_fill(-inf)
masking on current_class, CrossEntropyLoss, -0.1*reduce_sim). G4 replicates
the deployment forward (evaluate_with_margin / _phase2_train_step) with the
sole deliberate difference that the vit forward keeps its graph. Because
the vit call is argument-identical in both paths, ONE vit forward per batch
serves all four losses: same samples, same augmentation, same dropout
masks, same model state (strictly stronger than re-running two forwards).

Structural expectations (pre-registered BEFORE looking at results; derived
from Models/global_prompt.py with prompt_key=True, use_prompt_mask=False,
top_k=1):
  S1  reduce_sim only involves prompt_key (prompt_norm) dotted with the
      no-grad cls_features -> G2[prompt value] == 0 exactly.
  S2  the CE paths only involve the SELECTED prompt values (topk index is
      non-differentiable) -> G1[prompt key] == 0 and G4[prompt key] == 0
      exactly. The key-block cosines are therefore structurally undefined;
      they are reported with a zero-grad flag, NOT as 'inconsistent'.
  S3  by linearity, G3 == G1 + G2 per block (numerically verified per
      batch; g3_linearity_err must stay < 1e-6 or the probe is buggy).
  S4  for the prompt VALUE block the decisive metric therefore reduces to
      cos(G1, G4) == cos(G3, G4); all four are still computed and reported
      as specified.
  If measurements contradict S1-S3, the implementation is wrong: fix the
  probe first, do NOT interpret the numbers.

Pre-registered decision thresholds (frozen before running; applied per
probed task on the prompt VALUE block, per client = mean over batches):
  T1  support JE design review:  >= 3/5 clients with
      mean cos(G1,G4) < 0.3 AND negative-cosine batch fraction >= 0.4
      (and G4 non-zero).
  T2  JE premise weak, hold:     >= 4/5 clients with
      mean cos(G1,G4) >= 0.7 AND negative-cosine fraction <= 0.1.
  T3  gradient blocked:          G4 value-block norm < 1e-12 on ALL batches
      for a client -> audit the forward chain (detach / freeze / routing)
      before any interpretation.
  Anything in between: discuss at design review. A low cosine proves the
  two objectives propose different local update directions for the shared
  prompt parameters; it does NOT by itself prove joint training helps.

Read-only guarantees:
  - torch.autograd.grad only; no optimizer, no .step(), no .backward()
    accumulation into .grad, no state mutation.
  - Parameter/buffer fingerprint (exact float64 sums + abs-sums over
    state_dicts of vit / Tail_Anchor / original_model) is taken after
    setup and verified UNCHANGED after every client's computation
    (hard RuntimeError on any drift).
  - All RNG streams (torch cpu/cuda, python, numpy) snapshot/restored
    around each client's probe; batches are cached once from a seeded
    DataLoader (seed 9700+100*cid+task, matching hard_example_probe) and
    the SAME tensors feed all four losses.
  - Uses only existing checkpoints and local TRAIN-split data (never the
    test split); Client_DF training logic untouched.

Outputs (under {run_dir}/metrics/ by default):
  je_gradient_probe.csv         one row per (task, client, batch, block)
  je_gradient_probe_summary.csv one row per (task, client, block)
  console: per-client summary + pooled stats + threshold adjudication

Usage (from repo root, on the server):
  python diagnostics/je_gradient_probe.py \
      --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
      --tasks 4,0
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diag_utils import bootstrap_server

EPS = 1e-12
G3_LINEARITY_TOL = 1e-6


def parse_args():
    p = argparse.ArgumentParser("A2-JE gradient-consistency probe")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tasks", default="4,0",
                   help="comma-separated task ids, each probed at its "
                        "task_XX_end.pth state (default: 4,0)")
    p.add_argument("--n_batches", type=int, default=8,
                   help="fixed training batches per (client, task); "
                        "persistent conflict vs single-batch noise is "
                        "separated by the per-batch rows (default: 8)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="local training batch size (protocol hardcodes "
                        "16 in Client_DF.train(); default: 16)")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# read-only plumbing
# ---------------------------------------------------------------------------

def rng_state_dict():
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng_state(state):
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda_all"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def state_fingerprint(modules):
    """Exact fingerprint of every state_dict entry (params + buffers):
    (module_tag, key, shape, dtype, float64 sum, float64 abs-sum)."""
    entries = []
    for tag, mod in modules:
        for key, t in mod.state_dict().items():
            td = t.detach().double()
            entries.append((tag, key, tuple(t.shape), str(t.dtype),
                            float(td.sum().item()),
                            float(td.abs().sum().item())))
    return entries


def assert_fingerprint_unchanged(before, after):
    if before != after:
        for b, a in zip(before, after):
            if b != a:
                raise RuntimeError(
                    "READ-ONLY VIOLATION: state changed during probe: "
                    f"{b[:4]} -> {a[:4]}")
        raise RuntimeError("READ-ONLY VIOLATION: state fingerprint changed")


# ---------------------------------------------------------------------------
# gradient helpers
# ---------------------------------------------------------------------------

def grad_vec(loss, params, retain=True):
    """Flattened gradient of `loss` w.r.t. `params` (None -> exact zeros,
    flagged by norm == 0)."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                allow_unused=True)
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p).flatten())
        else:
            parts.append(g.detach().flatten())
    return torch.cat(parts) if parts else torch.zeros(0)


def group_grad_norm(loss, params):
    """Combined L2 norm of grad w.r.t. a group of params (None -> 0)."""
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True,
                                allow_unused=True)
    s = 0.0
    for g in grads:
        if g is not None:
            s += float(g.detach().double().pow(2).sum().item())
    return math.sqrt(s)


def cos_sim(a, b):
    na, nb = float(a.double().norm().item()), float(b.double().norm().item())
    if na < EPS or nb < EPS:
        return float("nan")
    return float(torch.dot(a.double(), b.double()).item() / (na * nb))


def vec_norm(v):
    return float(v.double().norm().item())


# ---------------------------------------------------------------------------
# probe core
# ---------------------------------------------------------------------------

def probe_client_task(client, task, nb_classes, n_batches, batch_size):
    """G1-G4 for one (client, task) at its task-end state. Returns
    (per_batch_rows, n_batches_used)."""
    if client.original_model is None:
        raise RuntimeError("client.original_model is None; unexpected state")
    if client.task_id != task:
        print(f"[probe] client {client.id}: task_id={client.task_id} != "
              f"{task} - skipped (checkpoint not at task end?)")
        return [], 0
    if client.heads[task] is None:
        raise RuntimeError(f"client {client.id}: head for task {task} missing")
    if client.prompts is None:
        raise RuntimeError(
            f"client {client.id}: prompts is None at task end; refusing to "
            f"probe against random init_prompts")
    if client.traindata is None:
        raise RuntimeError(f"client {client.id}: traindata missing")

    device = client.device
    # --- state setup, mirroring Client_DF.train() exactly ---
    client.vit.load_prompts(client.prompts)        # train() L289-290
    client.model.load_head(client.heads[task])     # deployment Chead (task t)
    client.model.to(device)
    client.vit.to(device)
    client.original_model.eval()                   # train() L287
    client.vit.train()                             # official loop never flips
    client.model.train()                           # vit/model default modes

    # --- parameter inventory ---
    # prompt blocks (P): everything under vit's "prompt." submodule
    prompt_blocks = {}   # block_name -> [params]
    aux_groups = {       # other trainable params, recorded SEPARATELY
        "vit_head": [], "vit_fc_norm": [], "vit_other": [],
    }
    other_names = []
    for n, p in client.vit.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("prompt."):
            prompt_blocks.setdefault(n[len("prompt."):], []).append(p)
        elif n.startswith("head."):
            aux_groups["vit_head"].append(p)
        elif n.startswith("fc_norm."):
            aux_groups["vit_fc_norm"].append(p)
        else:
            aux_groups["vit_other"].append(p)
            other_names.append(n)
    if not prompt_blocks:
        raise RuntimeError("no trainable prompt parameters found in vit")
    if other_names:
        print(f"[probe] WARNING: unclassified trainable vit params: "
              f"{other_names}")
    deploy_aux = {
        "chead": list(client.model.head.parameters()),
        "tail_key": [client.model.key],
        "tail_anchor": [client.model.anchor_pool],
    }

    # --- read-only fingerprint (after ALL setup mutations) ---
    modules = [("vit", client.vit), ("tail_anchor", client.model),
               ("original", client.original_model)]
    fingerprint_before = state_fingerprint(modules)
    rng_snapshot = rng_state_dict()

    # --- fixed batches: cached once, reused by every loss ---
    torch.manual_seed(9700 + 100 * client.id + task)
    g = torch.Generator()
    g.manual_seed(9700 + 100 * client.id + task)
    loader = DataLoader(client.traindata, batch_size=batch_size,
                        num_workers=0, shuffle=True, generator=g)
    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append((x.to(device, non_blocking=True),
                        y.long().to(device, non_blocking=True)))
    if not batches:
        raise RuntimeError(f"client {client.id}: no training batches")

    # --- masking (identical to phase-1 L324-326) ---
    mask = client.current_class
    not_mask = np.setdiff1d(np.arange(nb_classes), mask)
    not_mask_t = torch.tensor(not_mask, dtype=torch.int64).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    rows = []
    try:
        for bidx, (input, target) in enumerate(batches):
            # single vit forward shared by BOTH paths (see docstring):
            # original_model under no_grad + vit WITH grad, train=True.
            with torch.no_grad():
                out_orig = client.original_model(input)
                cls_features = out_orig['pre_logits']
            res = client.vit(input, task_id=client.task_id,
                             cls_features=cls_features, train=True)

            # phase-1 losses (train() L320-329, verbatim structure)
            logits = res['logits']
            pull_off = res['reduce_sim']
            logits_m = logits.index_fill(dim=1, index=not_mask_t,
                                         value=float('-inf'))
            loss_ce = criterion(logits_m, target)
            loss_pull = -0.1 * pull_off
            loss_full = loss_ce + loss_pull

            # deployment loss (evaluate_with_margin forward, grad kept
            # through vit on purpose: that no_grad cut is the object of
            # study). Tail_Anchor's second arg is the official (unused)
            # class_mask slot; pass target exactly as phase-2 does.
            feat = res['feat'].to(device)
            pre, _mixed, _pull2 = client.model(feat, target)
            pre_m = pre.index_fill(dim=1, index=not_mask_t,
                                   value=float('-inf'))
            loss_dep = criterion(pre_m, target)

            per_block = {}
            for bname, params in prompt_blocks.items():
                g1 = grad_vec(loss_ce, params)
                g2 = grad_vec(loss_pull, params)
                g3 = grad_vec(loss_full, params)
                # retain_graph=True everywhere: the aux grads below still
                # need the shared graph (it is freed when this iteration's
                # loss tensors go out of scope)
                g4 = grad_vec(loss_dep, params)
                per_block[bname] = (g1, g2, g3, g4)

            # aux gradients (heads/keys recorded separately, never mixed
            # into the prompt cosines)
            aux = {
                "aux_g3_vit_head": group_grad_norm(
                    loss_full, aux_groups["vit_head"]),
                "aux_g3_vit_fc_norm": group_grad_norm(
                    loss_full, aux_groups["vit_fc_norm"]),
                "aux_g3_vit_other": group_grad_norm(
                    loss_full, aux_groups["vit_other"]),
                "aux_g4_chead": group_grad_norm(
                    loss_dep, deploy_aux["chead"]),
                "aux_g4_tail_key": group_grad_norm(
                    loss_dep, deploy_aux["tail_key"]),
                "aux_g4_tail_anchor": group_grad_norm(
                    loss_dep, deploy_aux["tail_anchor"]),
                "aux_g4_vit_fc_norm": group_grad_norm(
                    loss_dep, aux_groups["vit_fc_norm"]),
                "aux_g4_vit_other": group_grad_norm(
                    loss_dep, aux_groups["vit_other"]),
            }

            for bname, (g1, g2, g3, g4) in per_block.items():
                n1, n2 = vec_norm(g1), vec_norm(g2)
                n3, n4 = vec_norm(g3), vec_norm(g4)
                cos14, cos34 = cos_sim(g1, g4), cos_sim(g3, g4)
                cos12, cos13 = cos_sim(g1, g2), cos_sim(g1, g3)
                rows.append({
                    "task": task, "client": client.id, "batch": bidx,
                    "block": bname,
                    "g1_norm": n1, "g2_norm": n2,
                    "g3_norm": n3, "g4_norm": n4,
                    "cos_g1_g4": cos14, "cos_g3_g4": cos34,
                    "cos_g1_g2": cos12, "cos_g1_g3": cos13,
                    "ratio_g4_g3": n4 / (n3 + EPS),
                    "ratio_g2_g1": n2 / (n1 + EPS),
                    "neg_g1_g4": int(math.isfinite(cos14) and cos14 < 0),
                    "neg_g3_g4": int(math.isfinite(cos34) and cos34 < 0),
                    "g4_zero": int(n4 < 1e-12),
                    "g3_linearity_err": vec_norm(g3 - (g1 + g2)),
                    **aux,
                })
    finally:
        # read-only verification + RNG restoration, even on exception
        assert_fingerprint_unchanged(fingerprint_before,
                                     state_fingerprint(modules))
        restore_rng_state(rng_snapshot)

    return rows, len(batches)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = ["task", "client", "block", "n_batches",
                  "mean_g1_norm", "mean_g2_norm", "mean_g3_norm",
                  "mean_g4_norm", "mean_cos_g1_g4", "mean_cos_g3_g4",
                  "mean_cos_g1_g2", "mean_cos_g1_g3",
                  "frac_neg_g1_g4", "frac_neg_g3_g4",
                  "mean_ratio_g4_g3", "mean_ratio_g2_g1",
                  "frac_g4_zero", "max_g3_linearity_err"]


def summarize(rows):
    out = []
    by_key = {}
    for r in rows:
        by_key.setdefault((r["task"], r["client"], r["block"]), []).append(r)
    for (task, cid, block), rs in sorted(by_key.items()):
        n = len(rs)
        out.append({
            "task": task, "client": cid, "block": block, "n_batches": n,
            "mean_g1_norm": float(np.mean([r["g1_norm"] for r in rs])),
            "mean_g2_norm": float(np.mean([r["g2_norm"] for r in rs])),
            "mean_g3_norm": float(np.mean([r["g3_norm"] for r in rs])),
            "mean_g4_norm": float(np.mean([r["g4_norm"] for r in rs])),
            "mean_cos_g1_g4": float(np.nanmean(
                [r["cos_g1_g4"] for r in rs])),
            "mean_cos_g3_g4": float(np.nanmean(
                [r["cos_g3_g4"] for r in rs])),
            "mean_cos_g1_g2": float(np.nanmean(
                [r["cos_g1_g2"] for r in rs])),
            "mean_cos_g1_g3": float(np.nanmean(
                [r["cos_g1_g3"] for r in rs])),
            "frac_neg_g1_g4": float(np.mean([r["neg_g1_g4"] for r in rs])),
            "frac_neg_g3_g4": float(np.mean([r["neg_g3_g4"] for r in rs])),
            "mean_ratio_g4_g3": float(np.mean(
                [r["ratio_g4_g3"] for r in rs])),
            "mean_ratio_g2_g1": float(np.mean(
                [r["ratio_g2_g1"] for r in rs])),
            "frac_g4_zero": float(np.mean([r["g4_zero"] for r in rs])),
            "max_g3_linearity_err": float(np.max(
                [r["g3_linearity_err"] for r in rs])),
        })
    return out


def adjudicate(summary_rows):
    """Apply the pre-registered thresholds (docstring T1-T3) to the prompt
    VALUE block, per task."""
    verdicts = {}
    for task in sorted({r["task"] for r in summary_rows}):
        rows = [r for r in summary_rows
                if r["task"] == task and r["block"] == "prompt"]
        if not rows:
            verdicts[task] = "no prompt-value block found"
            continue
        n_clients = len({r["client"] for r in rows})
        blocked = [r for r in rows if r["frac_g4_zero"] == 1.0]
        t1 = [r for r in rows
              if r["mean_cos_g1_g4"] < 0.3 and r["frac_neg_g1_g4"] >= 0.4
              and r["frac_g4_zero"] == 0.0]
        t2 = [r for r in rows
              if r["mean_cos_g1_g4"] >= 0.7 and r["frac_neg_g1_g4"] <= 0.1]
        lin_ok = all(r["max_g3_linearity_err"] < G3_LINEARITY_TOL
                     for r in rows)
        if blocked:
            verdicts[task] = (f"T3 GRADIENT BLOCKED on client(s) "
                              f"{[r['client'] for r in blocked]} - audit "
                              f"forward chain before interpreting")
        elif len(t1) >= math.ceil(3 * n_clients / 5) and len(t1) >= 3:
            verdicts[task] = (f"T1 support JE design review "
                              f"({len(t1)}/{n_clients} clients: "
                              f"mean cos(G1,G4)<0.3 & neg-frac>=0.4)")
        elif len(t2) >= math.ceil(4 * n_clients / 5):
            verdicts[task] = (f"T2 JE premise weak - hold "
                              f"({len(t2)}/{n_clients} clients: "
                              f"mean cos(G1,G4)>=0.7 & neg-frac<=0.1)")
        else:
            verdicts[task] = ("between thresholds - take to design review "
                              f"(T1 clients: {len(t1)}, T2 clients: "
                              f"{len(t2)}, blocked: {len(blocked)})")
        if not lin_ok:
            verdicts[task] += (" | WARNING: G3 != G1+G2 beyond tolerance - "
                               "probe implementation suspect")
    return verdicts


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    tasks = [int(t) for t in args.tasks.split(",") if t.strip() != ""]

    all_rows = []
    for t in tasks:
        ckpt = run_dir / "checkpoints" / f"task_{t:02d}_end.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        print(f"[probe] bootstrapping task {t} state from {ckpt}")
        run_args, server = bootstrap_server(
            args.run_dir, checkpoint=str(ckpt),
            data_path=args.data_path, device=args.device)
        nb_classes = run_args.nb_classes
        for client in server.clients:
            rows, used = probe_client_task(client, t, nb_classes,
                                           args.n_batches, args.batch_size)
            if rows:
                n_blocks = len({r["block"] for r in rows
                                if r["client"] == client.id})
                print(f"[probe] client {client.id} task {t}: "
                      f"{used} batches x {n_blocks} prompt blocks done")
                all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("no probe rows produced")

    # ---- per-batch CSV ----
    out_csv = args.out_csv or str(run_dir / "metrics" / "je_gradient_probe.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved {out_csv} ({len(all_rows)} rows)")

    # ---- summary CSV ----
    summary = summarize(all_rows)
    sum_csv = str(Path(out_csv).with_name(
        Path(out_csv).stem + "_summary.csv"))
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    print(f"saved {sum_csv}")

    # ---- console report ----
    print("\n===== A2-JE gradient-consistency probe =====")
    for s in summary:
        print(f"task {s['task']} client {s['client']} block "
              f"{s['block']}: cos(G1,G4)={s['mean_cos_g1_g4']:+.3f} "
              f"cos(G3,G4)={s['mean_cos_g3_g4']:+.3f} "
              f"neg-frac(G1,G4)={s['frac_neg_g1_g4']:.2f} "
              f"|G4|/|G3|={s['mean_ratio_g4_g3']:.3f} "
              f"g4-zero={s['frac_g4_zero']:.2f} "
              f"|G2|/|G1|={s['mean_ratio_g2_g1']:.3f} "
              f"lin-err={s['max_g3_linearity_err']:.2e}")
    print("\n----- pre-registered adjudication (prompt VALUE block) -----")
    for task, verdict in adjudicate(summary).items():
        print(f"task {task}: {verdict}")
    print("\nNOTE: prompt_key block cosines are structurally undefined "
          "(G1=G4=0 there, see docstring S2); read the 'prompt' (value) "
          "block rows for the decision metrics.")


if __name__ == "__main__":
    main()
