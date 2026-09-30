"""Phase 1 diagnostic: spatial multimodality analysis.

For each public class c (present at >= 2 clients) computes, on the extracted
pre-anchor features (z_ref and/or z_op -- NEVER [z;anchor]):
  - client-class means mu_{i,c}, within/between-client scatter S_W / S_B, H_c
  - K=1 vs K=2 isotropic (spherical) GMM on an 80/20 diagnostic-fit /
    diagnostic-validation split of the TRAIN features:
      sse1 / sse2 / delta_sse   (SSE can never alone prove multimodality)
      nll1 / nll2 / delta_nll   (held-out NLL; delta_nll > 0 favors K=2)
      bic1 / bic2 / delta_bic   (p_K = K*d + K + (K-1); delta_bic > 0 favors K=2)
  - silhouette (K=2 hard assignment)
  - mode_distance ||mu1 - mu2|| and mode separation
        rho = ||mu1-mu2|| / sqrt(d * (sigma1^2 + sigma2^2) / 2)
  - mode1_mass / mode2_mass
  - single-prototype coverage distortion
        D_cover = sum_k pi_k ||mu_k - mu_single||^2,  mu_single = sum_k pi_k mu_k
  - client composition entropy per mode
        r_i,k = n_i,k / sum_j n_j,k ;  H_k = -sum_i r_i,k log r_i,k

Outputs:
  - multimodality_summary.csv   (default: alongside the features .npz)
  - multimodality_aggregate.json (mean / median / std / p25 / p75 /
    bootstrap 95% CI over classes, per feature type)
  - optional per-class PCA scatter plots (--plot_dir; visualization only,
    never used for metrics)

Usage (from repo root, on the server):
  python diagnostics/analyze_multimodality.py \
      --features diagnostics_output/features_cifar100_task_04_end.npz
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

METRICS_FOR_AGGREGATE = [
    "delta_sse", "delta_nll", "delta_bic", "silhouette", "mode_distance",
    "rho", "mode1_mass", "coverage_distortion", "client_entropy_mode1",
    "client_entropy_mode2", "H_c",
]


def parse_args():
    p = argparse.ArgumentParser("Phase-1 multimodality analysis")
    p.add_argument("--features", required=True,
                   help="features .npz produced by diagnostics/extract_features.py")
    p.add_argument("--out_csv", default=None,
                   help="output summary CSV (default: <features dir>/multimodality_summary.csv)")
    p.add_argument("--aggregate_json", default=None,
                   help="aggregate JSON (default: alongside out_csv, multimodality_aggregate.json)")
    p.add_argument("--feature", default="both", choices=["z_ref", "z_op", "both"],
                   help="which feature set to analyze (default: both)")
    p.add_argument("--split", default="train", choices=["train", "test", "all"],
                   help="which split to use (default: train; modes must be discovered on TRAIN only)")
    p.add_argument("--all_classes", action="store_true",
                   help="analyze all classes instead of public classes only")
    p.add_argument("--gmm_seed", default=0, type=int)
    p.add_argument("--heldout_seed", default=0, type=int)
    p.add_argument("--boot_seed", default=0, type=int)
    p.add_argument("--n_boot", default=1000, type=int,
                   help="bootstrap resamples for the 95% CI")
    p.add_argument("--min_samples", default=20, type=int,
                   help="skip classes with fewer samples")
    p.add_argument("--plot_dir", default=None,
                   help="optional: save per-class PCA scatter plots here (visualization only)")
    return p.parse_args()


def class_scatter(z_by_client):
    """S_W, S_B, H_c per the Phase-1 formulas."""
    mus = {i: z.mean(axis=0) for i, z in z_by_client.items()}
    counts = {i: len(z) for i, z in z_by_client.items()}
    N = sum(counts.values())
    mu_c = sum(mus[i] * counts[i] for i in mus) / N
    S_W = sum(((z - mus[i]) ** 2).sum() for i, z in z_by_client.items()) / N
    S_B = sum(counts[i] * ((mus[i] - mu_c) ** 2).sum()
              for i in mus) / N
    return S_W, S_B, S_B / (S_W + 1e-12), N, mu_c


def fit_stats(z, client_ids_c, gmm_seed, heldout_seed):
    """K=1 vs K=2 isotropic GMM + all per-class multimodality statistics.

    GMMs are fitted on the 80% diagnostic-fit part only; BIC uses the fit
    part (N = n_fit), NLL uses the 20% diagnostic-validation part."""
    n, d = z.shape
    rng = np.random.RandomState(heldout_seed)
    perm = rng.permutation(n)
    n_hold = max(1, int(0.2 * n))
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
    if len(fit_idx) < 4:
        hold_idx, fit_idx = perm, perm

    out = {"n_fit": int(len(fit_idx)), "n_val": int(len(hold_idx))}
    gms = {}
    for K in (1, 2):
        gm = GaussianMixture(n_components=K, covariance_type="spherical",
                             random_state=gmm_seed, n_init=3,
                             reg_covar=1e-6, max_iter=200)
        gm.fit(z[fit_idx])
        gms[K] = gm
        # sklearn spherical p_K = (K-1) weights + K*d means + K variances
        #                                 = K*d + K + (K-1)   (matches the spec)
        out[f"bic{K}"] = float(gm.bic(z[fit_idx]))
        out[f"nll{K}"] = float(-gm.score(z[hold_idx]))
    out["delta_bic"] = out["bic1"] - out["bic2"]   # > 0 favors K=2
    out["delta_nll"] = out["nll1"] - out["nll2"]   # > 0 favors K=2

    # SSE on all class samples of this split
    mu1_all = z.mean(axis=0)
    sse1 = float(((z - mu1_all) ** 2).sum())
    gm2 = gms[2]
    lab = gm2.predict(z)                            # hard K=2 assignment
    mu_k = gm2.means_                               # (2, d)
    sse2 = float(((z - mu_k[lab]) ** 2).sum())
    out["sse1"], out["sse2"] = sse1, sse2
    out["delta_sse"] = (sse1 - sse2) / (sse1 + 1e-12)

    if len(np.unique(lab)) >= 2:
        out["silhouette"] = float(silhouette_score(z, lab))
    else:
        out["silhouette"] = float("nan")

    # mode geometry
    v1, v2 = float(gm2.covariances_[0]), float(gm2.covariances_[1])
    pi1, pi2 = float(gm2.weights_[0]), float(gm2.weights_[1])
    md = float(np.linalg.norm(mu_k[0] - mu_k[1]))
    out["mode_distance"] = md
    out["rho"] = float(md / np.sqrt(d * (v1 + v2) / 2.0 + 1e-12))
    out["mode1_mass"], out["mode2_mass"] = pi1, pi2

    # single-prototype coverage distortion
    mu_single = pi1 * mu_k[0] + pi2 * mu_k[1]
    out["mu_single"] = mu_single                    # kept for plotting
    out["mode_centers"] = mu_k
    out["coverage_distortion"] = float(
        pi1 * ((mu_k[0] - mu_single) ** 2).sum()
        + pi2 * ((mu_k[1] - mu_single) ** 2).sum())

    # client composition entropy per mode
    for k in (0, 1):
        counts = {}
        for cid, l in zip(client_ids_c.tolist(), lab.tolist()):
            if l == k:
                counts[cid] = counts.get(cid, 0) + 1
        tot = sum(counts.values())
        if tot > 0:
            r = np.array([v / tot for v in counts.values()])
            out[f"client_entropy_mode{k + 1}"] = float(-(r * np.log(r)).sum())
        else:
            out[f"client_entropy_mode{k + 1}"] = float("nan")
    return out


def aggregate(values, boot_seed, n_boot):
    """mean / median / std / p25 / p75 / bootstrap-95%-CI of the mean."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)],
                   dtype=float)
    if len(v) == 0:
        return None
    res = {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
    }
    if len(v) > 1 and n_boot > 0:
        rng = np.random.RandomState(boot_seed)
        boots = np.array([v[rng.randint(0, len(v), len(v))].mean()
                          for _ in range(n_boot)])
        res["ci95_low"] = float(np.percentile(boots, 2.5))
        res["ci95_high"] = float(np.percentile(boots, 97.5))
    else:
        res["ci95_low"] = res["ci95_high"] = res["mean"]
    return res


def plot_class(out_png, z, client_ids_c, task_ids_c, stats):
    """PCA 2-D scatter for ONE class: color = client, marker = task,
    star = FedTA single prototype, cross = K=2 mode centers.
    Visualization only -- never used for metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # project features and the recorded 768-d centers with the SAME PCA
    pca = PCA(n_components=2, random_state=0).fit(z)
    z2 = pca.transform(z)
    mu_single2 = pca.transform(stats["mu_single"][None, :])[0]
    mu_k2 = pca.transform(stats["mode_centers"])

    clients = sorted(set(int(i) for i in client_ids_c))
    tasks = sorted(set(int(t) for t in task_ids_c))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(clients))))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for ci, c in enumerate(clients):
        for ti, t in enumerate(tasks):
            m = (client_ids_c == c) & (task_ids_c == t)
            if not m.any():
                continue
            ax.scatter(z2[m, 0], z2[m, 1], s=14,
                       color=colors[ci % len(colors)],
                       marker=markers[ti % len(markers)],
                       label=f"client{c}" if ti == 0 else None)
    ax.scatter(*mu_single2, marker="*", s=260, color="red",
               edgecolors="darkred", zorder=5, label="single proto")
    ax.scatter(mu_k2[:, 0], mu_k2[:, 1], marker="x", s=120, color="black",
               linewidths=2.5, zorder=5, label="K2 modes")
    ax.set_title(f"PCA (viz only): class scatter")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    data = np.load(args.features)
    meta_path = Path(args.features).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) \
        if meta_path.is_file() else {}

    labels = data["labels"]
    client_ids = data["client_ids"]
    task_ids = data["task_ids"]
    splits = data["splits"]

    if args.split == "train":
        keep = splits == 0
    elif args.split == "test":
        keep = splits == 1
    else:
        keep = np.ones(len(labels), dtype=bool)

    if args.all_classes:
        target_classes = sorted(int(c) for c in np.unique(labels[keep]))
    else:
        target_classes = meta.get("public_classes") or sorted(
            int(c) for c in np.unique(labels[keep])
            if len(np.unique(client_ids[keep & (labels == c)])) >= 2)

    feature_types = (["z_ref", "z_op"] if args.feature == "both"
                     else [args.feature])

    feat_dir = Path(args.features).parent
    out_csv = args.out_csv or str(feat_dir / "multimodality_summary.csv")
    agg_json = args.aggregate_json or str(
        Path(out_csv).with_name("multimodality_aggregate.json"))
    plot_dir = Path(args.plot_dir) if args.plot_dir else None
    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ft in feature_types:
        feats = data[ft]
        for c in target_classes:
            sel = keep & (labels == c)
            if sel.sum() < args.min_samples:
                continue
            z_by_client = {
                int(i): feats[sel & (client_ids == i)]
                for i in np.unique(client_ids[sel])}
            if len(z_by_client) < 2:
                continue  # public classes only: need >= 2 clients
            S_W, S_B, H_c, N, mu_c = class_scatter(z_by_client)
            z = feats[sel]
            row = {
                "feature_type": ft, "class_id": c,
                "n_clients": len(z_by_client), "sample_num": int(N),
                "S_W": S_W, "S_B": S_B, "H_c": H_c,
            }
            stats = fit_stats(z, client_ids[sel], args.gmm_seed, args.heldout_seed)
            for k, v in stats.items():
                if k not in ("mu_single", "mode_centers"):
                    row[k] = v
            rows.append(row)
            print(f"[{ft}] class {c:3d}: N={N:5d} clients={len(z_by_client)} "
                  f"H_c={H_c:.4f} dBIC={row['delta_bic']:.1f} "
                  f"dNLL={row['delta_nll']:.4f} rho={row['rho']:.2f} "
                  f"D_cover={row['coverage_distortion']:.2f}")
            if plot_dir is not None:
                try:
                    plot_class(plot_dir / f"{ft}_class_{c:03d}.png",
                               z, client_ids[sel], task_ids[sel], stats)
                except Exception as e:  # plots must never kill the analysis
                    print(f"  [plot skipped] {e}")

    if not rows:
        print("no class had enough samples / clients; nothing to write")
        return

    fields = ["feature_type", "class_id", "n_clients", "sample_num",
              "sse1", "sse2", "delta_sse", "n_fit", "n_val",
              "nll1", "nll2", "delta_nll", "bic1", "bic2", "delta_bic",
              "silhouette", "mode_distance", "rho",
              "mode1_mass", "mode2_mass", "coverage_distortion",
              "client_entropy_mode1", "client_entropy_mode2",
              "S_W", "S_B", "H_c"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # aggregate over classes, per feature type
    agg = {"features_file": str(args.features),
           "split": args.split,
           "metrics": METRICS_FOR_AGGREGATE,
           "per_feature": {}}
    for ft in feature_types:
        sub = [r for r in rows if r["feature_type"] == ft]
        if not sub:
            continue
        entry = {
            "n_classes": len(sub),
            "n_delta_bic_positive": int(sum(r["delta_bic"] > 0 for r in sub)),
            "n_delta_nll_positive": int(sum(r["delta_nll"] > 0 for r in sub)),
        }
        for m in METRICS_FOR_AGGREGATE:
            a = aggregate([r.get(m) for r in sub], args.boot_seed, args.n_boot)
            if a is not None:
                entry[m] = a
        agg["per_feature"][ft] = entry

    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # console summary
    for ft in feature_types:
        sub = [r for r in rows if r["feature_type"] == ft]
        if not sub:
            continue
        print(f"\n===== {ft}: {len(sub)} classes =====")
        print(f"classes with delta_bic > 0 (K=2 favored): "
              f"{sum(r['delta_bic'] > 0 for r in sub)}/{len(sub)}")
        print(f"classes with delta_nll > 0 (K=2 favored): "
              f"{sum(r['delta_nll'] > 0 for r in sub)}/{len(sub)}")
        print(f"mean rho = {np.mean([r['rho'] for r in sub]):.3f}, "
              f"mean D_cover = "
              f"{np.mean([r['coverage_distortion'] for r in sub]):.3f}")
    print(f"\nsaved {out_csv}")
    print(f"saved {agg_json}")
    if plot_dir is not None:
        print(f"plots in {plot_dir}")


if __name__ == "__main__":
    main()
