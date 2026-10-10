"""Phase 4 D1 supplement (offline): landing-spot analysis — where do OLD
features actually go under the CURRENT key, and does that explain the
relative damage? Tests the concentrated-collapse / attractor hypothesis
against the (already-refuted) old-support-exposure story.

Background (roadmap 6.0): D1-lite falsified the simple chain
"future traffic on the OLD SUPPORT set -> damage" (E_future had NO
predictive power under FE). The surviving hypothesis is that key drift
RE-ROUTES old features elsewhere (R_oc landings) and the damage happens
AT THE LANDING SLOTS — anchors there serve later classes (or are
untrained noise). This script measures the landing side, which D1-lite
could not (the _oc matrices had not been saved).

Per (client, old task t, class c) — ZERO forward passes, consumes the
D0 npz (own-era B_t, landing B_t+_oc under the FINAL key, accumulated
B_all) and the per-class damage CSV (D1-lite local output):

  p_own(j|c)        own-era routing distribution        (B_t row c)
  p_land(j|c)       landing distribution                (B_oc row c —
                    SAME cached samples, current key: routing change is
                    attributable to model state ONLY)
  modal_preserve    p_land(own modal slot) — all-or-none unit
  support_preserve  sum of p_land over the own-era support set
  H_landing         landing-mass share of the client's future traffic:
                    sum_j p_land(j) * F_later(j) / F_later.sum()
                    (same normalization as D1's H_weighted on the
                    support — directly comparable contrast pair)
  landing_modal_slot / owner class / owner share
  on_noise_slot     landing modal slot >= n_classes (untrained region:
                    the anchor there is init noise)

Pre-registered reads (descriptive; causal claims stay with A2):
  R1  H_landing vs KA_rel within (client,task) cells — attractor version
      of the refuted E_future. Landing exposure with predictive power
      where support exposure had none = landing story confirmed.
  R2  support_preserve vs KA_rel within cells (protection factor)
  R3  all-or-none: for strongly-bound classes (own modal_share > 0.8)
      modal_preserve should be bimodal (U-shaped), not centered
  R4  cross-class collapse per client: old classes' landing modal slots
      should concentrate on fewer slots than own-era modal slots

Usage (repo root; local, ~seconds):
  python diagnostics/analyze_landing_spots.py \
      --binding_npz diagnostics_output/slot_binding_matrices.npz \
      --damage_csv diagnostics_output/collision_exposure_local.csv \
      --out_csv diagnostics_output/landing_spots.csv \
      --out_json diagnostics_output/landing_spots.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args():
    p = argparse.ArgumentParser("Phase-4 landing-spot analysis (offline)")
    p.add_argument("--binding_npz",
                   default="diagnostics_output/slot_binding_matrices.npz")
    p.add_argument("--damage_csv",
                   default="diagnostics_output/collision_exposure_local.csv")
    p.add_argument("--out_csv", default="diagnostics_output/landing_spots.csv")
    p.add_argument("--out_json",
                   default="diagnostics_output/landing_spots.json")
    return p.parse_args()


def entropy_bits(p):
    q = p[p > 0]
    return float(-np.sum(q * np.log2(q))) if len(q) else 0.0


def main():
    args = parse_args()
    npz = np.load(args.binding_npz)
    dmg = pd.read_csv(args.damage_csv)
    if "old_old" in dmg.columns:
        dmg = dmg.rename(columns={"old_old": "oo"})
    dmg["KA_rel"] = dmg.KA / dmg.oo.replace(0, np.nan)
    dmg["total_rel"] = dmg.total / dmg.oo.replace(0, np.nan)

    # index the npz: which (client, task) matrices exist
    per_client_tasks = {}
    for k in npz.files:
        if "_task_" in k and not k.endswith(("_oc", "_co")):
            cid, t = int(k.split("_")[1]), int(k.split("_")[3])
            per_client_tasks.setdefault(cid, []).append(t)
    for cid in per_client_tasks:
        per_client_tasks[cid].sort()

    # class -> own task map (row sums of B_t are nonzero only for its
    # task's classes; every sample contributes exactly one hit)
    task_of = {}
    n_classes = npz["client_0_all"].shape[0]
    for cid, tasks in per_client_tasks.items():
        for t in tasks:
            B_t = npz[f"client_{cid}_task_{t}"]
            for c in np.nonzero(B_t.sum(axis=1))[0]:
                task_of[(cid, int(c))] = t

    rows = []
    for (cid, c), t in sorted(task_of.items()):
        last = per_client_tasks[cid][-1]
        if t >= last:
            continue  # old classes only: no damage join, no future
        B_t = npz[f"client_{cid}_task_{t}"]
        B_oc = npz[f"client_{cid}_task_{t}_oc"]
        B_all = npz[f"client_{cid}_all"]
        n = B_t[c].sum()
        if n == 0:
            continue
        p_own = B_t[c].astype(float) / n
        p_land = B_oc[c].astype(float) / n
        supp = np.nonzero(B_t[c])[0]

        m_own = int(p_own.argmax())
        modal_share = float(p_own[m_own])
        m_land = int(p_land.argmax())

        # future (later-task) traffic per slot, at the later classes' own
        # task-end eras — same convention as D1's E_future/H_weighted
        F_later = np.zeros(B_t.shape[1])
        later_classes = []
        for t2 in per_client_tasks[cid]:
            if t2 > t:
                B2 = npz[f"client_{cid}_task_{t2}"]
                F_later += B2.sum(axis=0)
                later_classes += [int(x) for x in
                                  np.nonzero(B2.sum(axis=1))[0]]
        F_tot = F_later.sum()

        support_preserve = float(p_land[supp].sum())
        modal_preserve = float(p_land[m_own])
        h_landing = float(np.sum(p_land * F_later / F_tot)) if F_tot > 0 else 0.0
        h_support = float(np.sum(p_own[supp] * F_later[supp] / F_tot)) \
            if F_tot > 0 else 0.0

        # who dominates the landing modal slot (later classes only)?
        col = B_all[later_classes, m_land] if later_classes \
            else np.zeros(1)
        owner = int(later_classes[int(col.argmax())]) if len(col) else -1
        col_tot = float(B_all[:, m_land].sum())
        owner_share = float(col.max() / col_tot) if col_tot > 0 else 0.0
        future_mass_on_landing = float(F_later[m_land]) / F_tot \
            if F_tot > 0 else 0.0

        rows.append({
            "client_id": cid, "task": t, "class_id": c, "n_train": int(n),
            "modal_slot": m_own, "modal_share": modal_share,
            "landing_modal_slot": m_land,
            "landing_modal_share": float(p_land[m_land]),
            "landing_entropy": entropy_bits(p_land),
            "modal_preserve": modal_preserve,
            "support_preserve": support_preserve,
            "H_landing": h_landing, "H_support": h_support,
            "landing_owner_class": owner,
            "landing_owner_share": owner_share,
            "future_mass_on_landing_modal": future_mass_on_landing,
            "on_noise_slot": int(m_land >= n_classes),
        })

    df = pd.DataFrame(rows).merge(
        dmg[["client_id", "task", "class_id", "scope", "n_test",
             "oo", "KA", "P", "total", "KA_rel", "total_rel"]],
        on=["client_id", "task", "class_id"], how="inner")
    if df.empty:
        raise RuntimeError("empty join — check damage CSV coverage")
    w = df.n_test

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows -> {out_csv}")

    summary = {"n_classes": len(df)}

    def wmean(col):
        d = df.dropna(subset=[col])
        ww = d.n_test
        return float((d[col] * ww).sum() / ww.sum())

    print("\n===== landing descriptive stats (n_test-weighted) =====")
    for col in ["landing_modal_share", "landing_entropy", "modal_preserve",
                "support_preserve", "H_landing", "H_support",
                "future_mass_on_landing_modal"]:
        summary[f"wmean_{col}"] = wmean(col)
        print(f"  {col:>28s}: {wmean(col):.4f}")
    noise_share = float(df.on_noise_slot.mean())
    summary["landing_modal_on_noise_slot"] = noise_share
    print(f"  classes landing modal slot on UNTRAINED (>= {n_classes}) "
          f"region: {noise_share:.1%}")

    # ---------------- R1/R2: within-cell FE correlations ----------------
    print("\n===== within-(client,task) FE correlations vs KA_rel =====")
    print("  (pooled Spearman shown only as confounded reference)")
    def fe_sp(x, y):
        d = df[[x, y, "client_id", "task"]].dropna()
        d = d.assign(
            dx=d[x] - d.groupby(["client_id", "task"])[x].transform("mean"),
            dy=d[y] - d.groupby(["client_id", "task"])[y].transform("mean"))
        cells = d.groupby(["client_id", "task"]).size()
        d = d[d.groupby(["client_id", "task"])[x].transform("size") >= 5]
        if len(d) < 5:
            return float("nan"), float("nan")
        r_, p_ = spearmanr(d.dx, d.dy)
        return float(r_), float(p_)

    def pooled_sp(x, y):
        d = df[[x, y]].dropna()
        if len(d) < 5:
            return float("nan"), float("nan")
        r_, p_ = spearmanr(d[x], d[y])
        return float(r_), float(p_)

    for x in ["H_landing", "H_support", "modal_preserve",
              "support_preserve", "landing_modal_share",
              "future_mass_on_landing_modal"]:
        r_fe, p_fe = fe_sp(x, "KA_rel")
        r_pl, p_pl = pooled_sp(x, "KA_rel")
        summary[f"FE_rho_{x}_vs_KA_rel"] = r_fe
        summary[f"FE_p_{x}_vs_KA_rel"] = p_fe
        summary[f"pooled_rho_{x}_vs_KA_rel"] = r_pl
        print(f"  {x:>28s}: FE rho={r_fe:+.3f} (p={p_fe:.4g})"
              f"   [pooled {r_pl:+.3f}]")

    print("\n  same reads against total_rel:")
    for x in ["H_landing", "modal_preserve", "support_preserve"]:
        r_fe, p_fe = fe_sp(x, "total_rel")
        summary[f"FE_rho_{x}_vs_total_rel"] = r_fe
        summary[f"FE_p_{x}_vs_total_rel"] = p_fe
        print(f"  {x:>28s}: FE rho={r_fe:+.3f} (p={p_fe:.4g})")

    # contrast pair: landing exposure vs support exposure (the R1 verdict)
    r1, p1 = fe_sp("H_landing", "KA_rel")
    r0, p0 = fe_sp("H_support", "KA_rel")
    summary["R1_landing_vs_support"] = {
        "FE_rho_H_landing": r1, "FE_p_H_landing": p1,
        "FE_rho_H_support": r0, "FE_p_H_support": p0}

    # ---------------- R3: all-or-none bimodality ----------------
    print("\n===== R3: all-or-none (strongly-bound classes, modal_share>0.8) =====")
    strong = df[df.modal_share > 0.8]
    lo = float((strong.modal_preserve < 0.2).mean())
    mid = float(((strong.modal_preserve >= 0.2) &
                (strong.modal_preserve < 0.8)).mean())
    hi = float((strong.modal_preserve >= 0.8).mean())
    w_strong = float(strong.n_test.sum() / df.n_test.sum())
    summary["R3_strong_classes"] = {
        "n": len(strong), "mass_share": w_strong,
        "modal_preserve<0.2": lo, "0.2-0.8": mid, ">=0.8": hi}
    print(f"  n={len(strong)} ({w_strong:.0%} of test mass): "
          f"preserve<0.2: {lo:.1%}   0.2-0.8: {mid:.1%}   >=0.8: {hi:.1%}")
    print("  -> U-shape (two ends dominant) supports all-or-none collapse;"
          "\n     a mid-centered distribution supports gradual decay")

    # damage conditional on preserve for strong classes (all-or-none
    # implies the LOST arm carries the damage)
    for lab, m_ in [("preserve<0.2", strong.modal_preserve < 0.2),
                    ("preserve>=0.8", strong.modal_preserve >= 0.8)]:
        sub = strong[m_]
        if len(sub) >= 5:
            ww = sub.n_test
            ka = float((sub.KA_rel * ww).sum() / ww.sum())
            summary[f"R3_strong_{lab}_KA_rel"] = ka
            print(f"  strong & {lab}: n={len(sub)}  wKA_rel={ka:.3f}")

    # ---------------- R4: cross-class collapse ----------------
    print("\n===== R4: cross-class landing collapse (per client, old tasks) =====")
    collapse_stats = {}
    for cid, tasks in sorted(per_client_tasks.items()):
        old_tasks = [t for t in tasks if t < tasks[-1]]
        own_modals, land_modals, land_mass = set(), set(), np.zeros(
            npz[f"client_{cid}_all"].shape[1])
        L = np.zeros_like(land_mass)
        for t in old_tasks:
            B_t = npz[f"client_{cid}_task_{t}"]
            B_oc = npz[f"client_{cid}_task_{t}_oc"]
            for c in np.nonzero(B_t.sum(axis=1))[0]:
                own_modals.add(int(B_t[c].argmax()))
                land_modals.add(int(B_oc[c].argmax()))
                p_land = B_oc[c].astype(float) / B_oc[c].sum()
                L += p_land
        # landing-mass concentration: how many slots hold >=10% of a
        # class's landing mass, summed over classes; and the top slots
        top_slots = np.argsort(-L)[:10]
        top_share = float(L[top_slots].sum() / L.sum())
        n_old = sum(int((npz[f"client_{cid}_task_{t}"].sum(axis=1) > 0).sum())
                    for t in old_tasks)
        stats = {
            "n_old_classes": n_old,
            "distinct_own_modal_slots": len(own_modals),
            "distinct_landing_modal_slots": len(land_modals),
            "top10_landing_slot_mass_share": top_share,
            "top_landing_slots": [int(j) for j in top_slots[:5]],
        }
        collapse_stats[cid] = stats
        print(f"  c{cid}: {n_old} old classes: own modal slots="
              f"{len(own_modals)}  landing modal slots={len(land_modals)}"
              f"  top-10 landing mass={top_share:.1%}"
              f"  top={stats['top_landing_slots']}")
    summary["R4_collapse_per_client"] = collapse_stats

    # public/private stratification
    print("\n===== public/private stratification (n_test-weighted) =====")
    for scope, sub in df.groupby("scope"):
        ww = sub.n_test
        print(f"  {scope}: n={len(sub)}  "
              f"wKA_rel={float((sub.KA_rel*ww).sum()/ww.sum()):.3f}  "
              f"wH_landing={float((sub.H_landing*ww).sum()/ww.sum()):.4f}  "
              f"w_modal_preserve="
              f"{float((sub.modal_preserve*ww).sum()/ww.sum()):.3f}  "
              f"w_support_preserve="
              f"{float((sub.support_preserve*ww).sum()/ww.sum()):.3f}")
        r_fe, p_fe = fe_sp("H_landing", "KA_rel")
        summary[f"{scope}_FE_rho_H_landing_vs_KA_rel"] = r_fe
        summary[f"{scope}_FE_p_H_landing_vs_KA_rel"] = p_fe
        print(f"    FE rho(H_landing, KA_rel) pooled-by-scope={r_fe:+.3f}"
              f" (p={p_fe:.4g})")

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary -> {args.out_json}")
    print("NOTE: descriptive landing audit — causal attribution still "
          "requires the A2 intervention.")


if __name__ == "__main__":
    main()
