"""Phase 4 D1 v2 (offline): collision exposure -> interface displacement
-> KA damage. ZERO forward passes — consumes D0 outputs, task_end
checkpoints (key/anchor tensors only) and the 4-combo per-class CSV.

Mechanism chain to test (per class c of an old task t):

  E_c (future collision exposure)  ->  D_c (weighted interface drift)
                                    ->  KA_c (accuracy damage oo-oc)

Quantities per (client, old task t, class c):

  p(j|c)      class c's own-era routing distribution (D0 matrix B_t row c)
  CEI_all     GPT-form contention: sum_j p(j|c) * (1 - q(c|j)),
              q from the client's full-sequence matrix B_all
  E_future    forward-looking exposure: sum_j p(j|c) * F_j where
              F_j = hits on slot j from classes of LATER tasks
              (each later class counted at its own task_end routing state)
  d_key_c     weighted key displacement sum_j p(j|c) * ||K_j^final - K_j^t||
  d_anchor_c  same for anchor_pool rows
  D_traffic   traffic-weighted displacement sum_j p(j|c) * (F_j/tot_future)
              * ||K_j^final - K_j^t|| — H_weighted extended with per-slot
              displacement; separates "hit" from "hit hard" where the
              binary F_j>0 split cannot
  hijack vs decay split: support slots split by F_j>0 vs F_j==0,
              weighted displacement of each subset — the A2 design fork
              (gradient mask vs freeze+wd-exemption) hangs on this
  modal reuse future traffic on c's MODAL slot (the "where did old
              interfaces go" sanity check for stabK=0.00)
  KA_c / P_c / total   joined from the 4-combo CSV (accuracy damage)

Usage (repo root, on the server; runs in ~1 min, no GPU forward):
  python diagnostics/collision_exposure_diagnostic.py \
      --run_dir output/cifar100/fedta/<run>/seed_42 \
      --task_ckpts checkpoints/task_00_end.pth,...,checkpoints/task_04_end.pth \
      --binding_csv diagnostics_output/slot_binding.csv \
      --binding_npz diagnostics_output/slot_binding_matrices.npz \
      --combo_csv diagnostics_output/class_retention_4combo.csv \
      --out_csv diagnostics_output/collision_exposure.csv \
      --out_json diagnostics_output/collision_exposure.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diag_utils  # noqa: F401 — inserts repo root into sys.path
from runtime_utils import load_checkpoint_file
from slot_collision_diagnostic import l2_normalize, resolve


def parse_args():
    p = argparse.ArgumentParser("Phase-4 D1 v2 offline collision exposure")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--task_ckpts", required=True,
                   help="comma-separated task_end checkpoints in task order")
    p.add_argument("--binding_csv", required=True)
    p.add_argument("--binding_npz", required=True)
    p.add_argument("--combo_csv", required=True)
    p.add_argument("--out_csv",
                   default="diagnostics_output/collision_exposure.csv")
    p.add_argument("--out_json",
                   default="diagnostics_output/collision_exposure.json")
    return p.parse_args()


def load_ka_tensors(ckpt_paths, client_ids):
    """key/anchor rows per client per boundary (cpu tensors, no model)."""
    out = []
    for path in ckpt_paths:
        ckpt = load_checkpoint_file(path)
        per_client = {}
        for cid in client_ids:
            ta = ckpt["clients"][cid]["tail_anchor_model"]
            per_client[cid] = {
                "key": ta["key"].detach().cpu().clone(),
                "anchor_pool": ta["anchor_pool"].detach().cpu().clone(),
            }
        out.append(per_client)
        del ckpt
    return out


def main():
    args = parse_args()
    ckpt_paths = [resolve(args.run_dir, p) for p in args.task_ckpts.split(",")]
    n_bounds = len(ckpt_paths)

    bind = pd.read_csv(args.binding_csv)
    npz = np.load(args.binding_npz)
    combo = pd.read_csv(args.combo_csv)

    # damage pivot from the 4-combo CSV: acc per (client, old_task, class,
    # prompt x key_anchor). NOTE: the 4-combo CSV stores acc as STRINGS
    # (analyze_class_retention.py writes f"{acc:.2f}") — cast to float.
    combo = combo.rename(columns={"old_task": "task"})
    combo["acc"] = combo["acc"].astype(float)
    pv = combo.pivot_table(
        index=["client_id", "task", "class_id", "scope", "n_test"],
        columns=["prompt", "key_anchor"], values="acc", aggfunc="first")
    # flatten the (prompt, key_anchor) MultiIndex manually — tuple-key
    # rename on MultiIndex columns is a pandas-version-dependent trap
    name_map = {("old", "old"): "oo", ("old", "current"): "oc",
                ("current", "old"): "co", ("current", "current"): "cc"}
    pv.columns = [name_map.get(tuple(c), c) for c in pv.columns]
    pv = pv.reset_index()
    missing = [c for c in ("oo", "oc", "co", "cc") if c not in pv.columns]
    if missing:
        raise RuntimeError(f"4-combo pivot missing combos {missing} — "
                           f"check the CSV's prompt/key_anchor values")
    pv["KA"] = pv["oo"] - pv["oc"]
    pv["P"] = pv["oo"] - pv["co"]
    pv["total"] = pv["oo"] - pv["cc"]
    damage = pv[["client_id", "task", "class_id", "scope", "n_test",
                 "oo", "KA", "P", "total"]]

    client_ids = sorted(bind.client_id.unique().tolist())
    ka = load_ka_tensors(ckpt_paths, client_ids)

    # class -> task map per client (D0 binding covers tasks 0..T-1)
    task_of = {}
    for r in bind.itertuples():
        task_of[(r.client_id, r.class_id)] = r.task

    rows = []
    for r in bind.itertuples():
        t = r.task
        if t >= n_bounds - 1:
            continue  # last task: no future, no damage join
        cid, c = r.client_id, r.class_id
        B_t = npz[f"client_{cid}_task_{t}"]
        B_all = npz[f"client_{cid}_all"]
        row = B_t[c].astype(np.float64)
        n_route = row.sum()
        if n_route == 0:
            continue
        p_j = row / n_route
        supp = np.nonzero(row)[0]

        later_classes = [cc for (cid2, cc), tt in task_of.items()
                         if cid2 == cid and tt > t]
        F_j = B_all[later_classes][:, supp].sum(axis=0) \
            if later_classes else np.zeros(len(supp))

        # CEI (GPT form): 1 - q(c|j) over the client's full sequence
        col_tot = B_all[:, supp].sum(axis=0)
        q_c = B_all[c, supp] / np.maximum(col_tot, 1e-12)
        cei_all = float(np.sum(p_j[supp] * (1.0 - q_c)))

        # forward-looking exposure
        e_future = float(np.sum(p_j[supp] * F_j))

        # review additions:
        # H_c — traffic-WEIGHTED hijack exposure (not binary F_j>0):
        #   share of the client's TOTAL future routing mass that lands on
        #   c's support, weighted by c's routing distribution
        tot_future = float(B_all[later_classes].sum()) if later_classes else 0.0
        h_weighted = float(np.sum(
            p_j[supp] * F_j / tot_future)) if tot_future > 0 else 0.0
        # CC_c — collision concentration: is there ONE dominant conflicted
        # interface?  (max_j p(j|c) * (1 - q(c|j)))
        cc_c = float(np.max(p_j[supp] * (1.0 - q_c)))
        # MHR_c — modal-slot hijack ratio: of all future traffic on c's
        # support, the share that hits the MODAL slot specifically
        modal = int(r.modal_slot)
        F_modal = float(B_all[later_classes, modal].sum()) if later_classes else 0.0
        mhr = F_modal / float(F_j.sum()) if F_j.sum() > 0 else 0.0

        # weighted interface displacement (t_end -> final)
        K_t = ka[t][cid]["key"]
        K_f = ka[n_bounds - 1][cid]["key"]
        A_t = ka[t][cid]["anchor_pool"]
        A_f = ka[n_bounds - 1][cid]["anchor_pool"]
        def wdisp(Ka, Kb):
            d = (Kb[supp] - Ka[supp]).norm(dim=tuple(range(1, Kb.dim())))
            return float(np.sum(p_j[supp] * d.numpy()))
        d_key = wdisp(K_t, K_f)
        d_anchor = wdisp(A_t, A_f)

        # traffic-weighted displacement: H_weighted extended with the
        # per-slot displacement — distinguishes "accessed" from
        # "accessed heavily" where the binary hit split cannot
        d_slots = (K_f[supp] - K_t[supp]).norm(
            dim=tuple(range(1, K_f.dim()))).numpy()
        d_traffic = float(np.sum(
            p_j[supp] * (F_j / tot_future) * d_slots)) \
            if tot_future > 0 else 0.0

        # direction/norm decomposition (D1 v2 FE verdict 2026-10-10: raw
        # ||dK|| is a BLIND metric — Tail_Anchor trains under COUPLED-wd
        # Adam (L2 into gradient), whose per-element adaptive scaling
        # sign-flips unhit rows toward zero: unhit slots random-walk in
        # DIRECTION while their norm collapses (~1e-2 of original), hit
        # slots keep norm but stay pinned — both produce comparable raw
        # displacement (hit/nohit ratio 0.91x). Routing sees direction
        # only (l2_normalize): d_key_dir is the eviction-relevant
        # quantity (FE rho vs modal_preserve -0.622), d_key_norm_ratio
        # is the wd-survival signature.)
        dims = tuple(range(1, K_f.dim()))
        u_t = l2_normalize(K_t[supp], dim=dims)
        u_f = l2_normalize(K_f[supp], dim=dims)
        cos = (u_t * u_f).sum(dim=dims).numpy()
        d_key_dir = float(np.sum(p_j[supp] * (1.0 - cos)))
        d_key_norm_ratio = float(np.sum(
            p_j[supp] * (K_f[supp].norm(dim=dims) /
                         K_t[supp].norm(dim=dims)).numpy()))

        # immediate next window (event-driven check)
        d_key_next = wdisp(K_t, ka[t + 1][cid]["key"]) \
            if t + 1 < n_bounds else np.nan

        # hijack vs decay split: slots with future traffic vs without
        hit_mask = F_j > 0
        if hit_mask.any() and not hit_mask.all():
            w_hit = p_j[supp][hit_mask]
            w_no = p_j[supp][~hit_mask]
            d_key_hit = float(np.sum(
                w_hit * (K_f[supp][hit_mask] - K_t[supp][hit_mask])
                .norm(dim=tuple(range(1, K_f.dim()))).numpy()))
            d_key_nohit = float(np.sum(
                w_no * (K_f[supp][~hit_mask] - K_t[supp][~hit_mask])
                .norm(dim=tuple(range(1, K_f.dim()))).numpy()))
            share_hit_mass = float(w_hit.sum())
        else:
            d_key_hit = d_key_nohit = np.nan
            share_hit_mass = float(hit_mask.mean())

        modal = int(r.modal_slot)
        modal_future_traffic = int(
            B_all[later_classes, modal].sum()) if later_classes else 0

        rows.append({
            "client_id": cid, "task": t, "class_id": c,
            "modal_slot": modal, "modal_share": r.modal_share,
            "CEI_all": cei_all, "E_future": e_future,
            "H_weighted": h_weighted, "CC": cc_c, "MHR": mhr,
            "D_traffic": d_traffic,
            "d_key": d_key, "d_anchor": d_anchor, "d_key_next": d_key_next,
            "d_key_dir": d_key_dir, "d_key_norm_ratio": d_key_norm_ratio,
            "d_key_hit_slots": d_key_hit, "d_key_nohit_slots": d_key_nohit,
            "hit_slot_mass_share": share_hit_mass,
            "modal_future_traffic": modal_future_traffic,
        })

    df = pd.DataFrame(rows).merge(
        damage, on=["client_id", "task", "class_id"], how="inner",
        suffixes=("", "_dmg"))
    if df.empty:
        raise RuntimeError("empty join — check binding/combo task coverage")

    # relative damage (D1-lite FE finding: absolute damage is dominated by
    # the baseline-accuracy ceiling, beta(oo)~0.77; relative KA/oo and
    # total/oo are the analyzable quantities)
    df["KA_rel"] = df.KA / df.oo.replace(0, np.nan)
    df["total_rel"] = df.total / df.oo.replace(0, np.nan)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows -> {out_csv}")

    # ---------------- descriptive chain audit ----------------
    from scipy.stats import spearmanr
    def sp(x, y):
        m = df[[x, y]].dropna()
        if len(m) < 5:
            return float("nan"), float("nan")
        r_, p_ = spearmanr(m[x], m[y])
        return float(r_), float(p_)

    summary = {"n_classes": len(df)}
    print(f"\nclass-units = {len(df)}")

    # coverage sanity (review check): classes per client per task from the
    # binding CSV vs manifest-implied 8/task; B_all total routed mass
    print("\n===== coverage sanity =====")
    for cid, sub in bind.groupby("client_id"):
        per_task = sub.groupby("task").size().tolist()
        print(f"  c{cid}: classes/task = {per_task}  "
              f"B_all mass = {int(npz[f'client_{cid}_all'].sum())}")
    expected = int(bind.groupby(["client_id", "task"]).size().mode().max())
    small = [f"c{r.client_id}t{r.task}" for r in bind.itertuples()
             if list(bind[bind.client_id == r.client_id].task ==
                     r.task).count(True) < expected]
    if small:
        print(f"  cells with fewer than {expected} classes: {small}")

    print("\n===== chain: exposure -> displacement -> damage =====")
    print("  (absolute damage is ceiling-dominated; KA_rel/total_rel are"
          " the meaningful quantities)")
    for a, b in [("E_future", "d_key"), ("H_weighted", "d_key"),
                 ("D_traffic", "d_key"),
                 ("d_key", "KA_rel"), ("d_anchor", "KA_rel"),
                 ("d_key_dir", "KA_rel"), ("d_key_norm_ratio", "KA_rel"),
                 ("d_key_dir", "d_key"),
                 ("E_future", "KA_rel"), ("H_weighted", "KA_rel"),
                 ("D_traffic", "KA_rel"),
                 ("CC", "KA_rel"), ("MHR", "KA_rel"),
                 ("d_key", "total_rel"), ("E_future", "total_rel"),
                 ("modal_share", "KA_rel"), ("modal_share", "total_rel"),
                 ("E_future", "KA"), ("CEI_all", "KA")]:
        r_, p_ = sp(a, b)
        summary[f"spearman_{a}_vs_{b}"] = r_
        summary[f"p_{a}_vs_{b}"] = p_
        print(f"  {a:>12s} vs {b:<8s}: rho={r_:+.3f}  p={p_:.4g}")

    print("\n===== hijack vs decay split (A2 design fork) =====")
    m = df.dropna(subset=["d_key_hit_slots", "d_key_nohit_slots"])
    if not m.empty:
        w = m.n_test
        wm_hit = float((m.d_key_hit_slots * w).sum() / w.sum())
        wm_no = float((m.d_key_nohit_slots * w).sum() / w.sum())
        summary["wmean_d_key_hit_slots"] = wm_hit
        summary["wmean_d_key_nohit_slots"] = wm_no
        print(f"  weighted d_key on slots WITH future traffic: {wm_hit:.4f}")
        print(f"  weighted d_key on slots WITHOUT future traffic: {wm_no:.4f}"
              f"   (ratio {wm_hit / max(wm_no, 1e-12):.2f}x)")

    print("\n===== modal-slot future reuse (stabK=0.00 sanity) =====")
    w = df.n_test
    wmr = float((df.modal_future_traffic * w).sum() / w.sum())
    reused = int((df.modal_future_traffic > 0).sum())
    summary["wmean_modal_future_traffic"] = wmr
    summary["modal_slots_reused_by_future"] = reused
    print(f"  classes whose modal slot receives future traffic: "
          f"{reused}/{len(df)}")
    print(f"  weighted mean future hits on modal slot: {wmr:.1f}")

    print("\n===== stratified: public vs private =====")
    for scope, sub in df.groupby("scope"):
        w = sub.n_test
        def wm(col):
            m2 = sub.dropna(subset=[col])
            w2 = m2.n_test
            return float((m2[col] * w2).sum() / w2.sum())
        r1, _ = sp_full = sp("E_future", "KA")
        r2, _ = sp("d_key", "KA")
        print(f"  {scope}: n={len(sub)}  wKA={wm('KA'):.2f}pp  "
              f"wE_future={wm('E_future'):.1f}  w_d_key={wm('d_key'):.4f}  "
              f"rho(E,KA)={r1:+.2f}  rho(d_key,KA)={r2:+.2f}")
        summary[f"{scope}_wKA"] = wm("KA")
        summary[f"{scope}_wE_future"] = wm("E_future")
        summary[f"{scope}_rho_E_vs_KA"] = r1

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary -> {args.out_json}")
    print("NOTE: descriptive audit — causal claims require the A2 "
          "intervention experiment.")


if __name__ == "__main__":
    main()
