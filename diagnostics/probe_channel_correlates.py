"""Sub-mechanism probe for the public-class KA excess (post 4-combo adjudication).

Consumes the 4-combo per-class CSV written by analyze_class_retention.py
(class_retention_4combo.csv) and tests two candidate sub-mechanisms:

  (i)  indirect co-adaptation: cross-client evolution of public-class
       representations co-opts the local key/anchor (aggregation-mediated,
       executing locally)  -> at matched local training strength, public
       KA damage exceeds private; KA_c weakly correlates with P_c
  (ii) protocol-structural fragility: Dirichlet-divided public classes are
       locally undertrained -> fragile slots  -> KA_c anti-correlates with
       n_test (fewer samples, more damage)

Verdict (CIFAR-100, seed 42, 158 class-units): (ii) REJECTED — the
correlation is POSITIVE (r=+0.346, spearman p<1e-4 within public): the
least-trained public classes (n_test<=15) are nearly immune (wKA=+1.8)
while entrenched public classes suffer most; (i) SUPPORTED in refined
form — at matched strength (woo~95) public exceeds private ~2x
(+19.7 vs +11.4 pooled; 4/5 clients) with co-elevated P channel
(+5.2 vs +2.3). Damage = interaction{local interface entrenchment} x
{federation-wide class-region evolution}, NOT "others retrain my class".

Usage (repo root; run anywhere with the CSV):
  python diagnostics/probe_channel_correlates.py \
      --csv diagnostics_output/class_retention_4combo.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args():
    p = argparse.ArgumentParser("channel-correlate probe on 4-combo CSV")
    p.add_argument("--csv", required=True, help="class_retention_4combo.csv")
    return p.parse_args()


def load_units(csv_path):
    df = pd.read_csv(csv_path)
    key = ["client_id", "old_task", "class_id"]
    base = df[key + ["scope", "n_clients", "n_test"]].drop_duplicates(key)
    for p_name, k_name, col in [("old", "old", "oo"),
                                ("old", "current", "oc"),
                                ("current", "old", "co"),
                                ("current", "current", "cc")]:
        sub = (df[(df.prompt == p_name) & (df.key_anchor == k_name)]
               [key + ["acc"]].rename(columns={"acc": col}))
        base = base.merge(sub, on=key, how="left")
    base["KA"] = base["oo"] - base["oc"]   # local key/anchor channel
    base["P"] = base["oo"] - base["co"]    # aggregated prompt channel
    return base.dropna(subset=["oo", "oc", "co", "cc"])


def wavg(g, c):
    return float(np.average(g[c], weights=g.n_test)) if len(g) else float("nan")


def main():
    args = parse_args()
    pv = load_units(args.csv)
    print(f"class-units = {len(pv)}")

    print("\n===== (ii) fragility test: KA_c vs n_test (expect <0 if true) "
          "=====")
    for scope, g in pv.groupby("scope"):
        r = pearsonr(g.n_test, g["KA"])[0]
        p = spearmanr(g.n_test, g["KA"])[1]
        print(f"{scope:7s} (n={len(g):3d}): r={r:+.3f}  spearman_p={p:.4f}")

    print("\n===== (i) co-adaptation test: KA_c vs P_c =====")
    for scope, g in pv.groupby("scope"):
        r = pearsonr(g["P"], g["KA"])[0]
        p = spearmanr(g["P"], g["KA"])[1]
        print(f"{scope:7s} (n={len(g):3d}): r={r:+.3f}  spearman_p={p:.4f}")

    print("\n===== entrenchment x evolution interaction (weighted, pp) =====")
    labels = ["1-15", "16-40", "41-80", "81-171"]
    pv["bucket"] = pd.cut(pv.n_test, [0, 15, 40, 80, 171], labels=labels)
    print(f"{'bucket':8s} {'n_pub':>5s} {'wKA_pub':>8s} {'woo_pub':>8s}")
    for lab in labels:
        g = pv[(pv.scope == "public") & (pv.bucket == lab)]
        if len(g):
            print(f"{lab:8s} {len(g):5d} {wavg(g, 'KA'):+8.2f} "
                  f"{wavg(g, 'oo'):8.2f}")
    gp = pv[pv.scope == "private"]
    print(f"private (all, n={len(gp)}): wKA={wavg(gp, 'KA'):+.2f}  "
          f"woo={wavg(gp, 'oo'):.2f}")

    print("\n===== matched-strength comparison: public n_test>=40 vs private "
          "=====")
    strong = pv[(pv.scope == "public") & (pv.n_test >= 40)]
    print(f"public entrenched: n={len(strong)}  wKA={wavg(strong, 'KA'):+.2f}  "
          f"wP={wavg(strong, 'P'):+.2f}  woo={wavg(strong, 'oo'):.2f}")
    print(f"private all     : n={len(gp)}  wKA={wavg(gp, 'KA'):+.2f}  "
          f"wP={wavg(gp, 'P'):+.2f}  woo={wavg(gp, 'oo'):.2f}")
    for cid, g in strong.groupby("client_id"):
        gpv = pv[(pv.scope == "private") & (pv.client_id == cid)]
        print(f"  c{cid}: public-strong wKA={wavg(g, 'KA'):+.2f} vs "
              f"private wKA={wavg(gpv, 'KA'):+.2f}")

    out = Path(args.csv).with_name("channel_correlates.json")
    import json
    report = {
        "n_units": int(len(pv)),
        "pearson_KA_ntest": {s: float(pearsonr(
            pv[pv.scope == s].n_test, pv[pv.scope == s]["KA"])[0])
            for s in ["public", "private"]},
        "pearson_KA_P": {s: float(pearsonr(
            pv[pv.scope == s]["P"], pv[pv.scope == s]["KA"])[0])
            for s in ["public", "private"]},
        "public_entrenched_n40_wKA": wavg(strong, "KA"),
        "private_all_wKA": wavg(gp, "KA"),
        "public_weak_le15_wKA": wavg(
            pv[(pv.scope == "public") & (pv.bucket == "1-15")], "KA"),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
