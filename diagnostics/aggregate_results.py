"""Formal-experiment results aggregation (PR-3, external review 2026-10-22).

Walks the unified output tree

    {out_root}/{dataset}/{method}/{run_name}/seed_{seed}/
        args.json (main repo) or meta.json (protocol adapters)
        metrics/summary_metrics.csv
        checkpoints/latest.pth (main repo, fedta_a5)

and emits the three-level (dataset / seed / task) long-format table the
project requires, PLUS the two PR-3 formal-pipeline guards:

  1. MIGRATION REJECTION (review §一/§三): any fedta_a5 run whose
     checkpoint carries the `a5_proto_migration` provenance marker is
     EXCLUDED from the aggregation (its prototype history is incomplete —
     a migration is a debug/compatibility mechanism, never a valid start
     for formal A5 results).
  2. ARM-IDENTITY VERIFICATION (review §三 "用实际 checkpoint 和日志验证
     实验臂身份"): for fedta_a5 runs the checkpoint's `a5_arm` descriptor
     (derived from ACTUAL runtime state) is cross-checked against
     args.json; a mismatch — or a missing descriptor (pre-PR-3 checkpoint)
     — excludes the run (arm identity unverifiable).

Both guards are fail-closed for fedta_a5: a run that cannot be verified
is rejected, never silently included.

Output:
  --out results.csv   long format, one row per accepted
                      (dataset, method, run_name, seed, after_task):
                      avg_accuracy / forgetting / bwt
  --report report.json  every scanned run with its accept/reject reason
  stdout              per-(dataset, method, run_name) seed-mean summary

Usage:
  python diagnostics/aggregate_results.py --roots output external_baselines/output \
      --out reports/formal_results.csv --report reports/formal_report.json
"""

import argparse
import csv
import json
from pathlib import Path

try:
    import torch

    def torch_load_compat(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch without the weights_only kwarg
            return torch.load(path, map_location="cpu")
except ImportError:  # torch-free environments: only non-A5 runs scannable
    def torch_load_compat(path):
        raise RuntimeError("torch is required to inspect fedta_a5 checkpoints")


def scan_run(run_dir):
    """Validate one seed-level run directory. Returns (rows, status) where
    status is ("accepted", None) or ("rejected", reason)."""
    run_dir = Path(run_dir)

    args_json = run_dir / "args.json"
    meta_json = run_dir / "meta.json"
    if args_json.is_file():
        with open(args_json, encoding="utf-8") as f:
            cfg = json.load(f)
    elif meta_json.is_file():
        with open(meta_json, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        return [], ("rejected", "no args.json / meta.json (not a run dir)")

    summary_csv = run_dir / "metrics" / "summary_metrics.csv"
    if not summary_csv.is_file():
        return [], ("rejected", "no metrics/summary_metrics.csv")

    method = str(cfg.get("method", ""))

    # ---- PR-3 formal-pipeline guards (fedta_a5 only) --------------------
    if method == "fedta_a5":
        latest = run_dir / "checkpoints" / "latest.pth"
        if not latest.is_file():
            return [], ("rejected",
                        "fedta_a5 run has no checkpoints/latest.pth — "
                        "migration marker / arm identity unverifiable")
        try:
            ckpt = torch_load_compat(latest)
        except Exception as e:  # unreadable checkpoint = unverifiable
            return [], ("rejected", f"checkpoint unreadable: {e}")

        # guard 1: migration provenance marker
        if "a5_proto_migration" in ckpt:
            return [], ("rejected",
                        "checkpoint carries the a5_proto_migration "
                        "provenance marker (incomplete prototype history; "
                        "explicitly excluded from formal results)")

        # guard 2: arm identity from ACTUAL checkpoint state vs args.json
        arm = ckpt.get("a5_arm")
        if not isinstance(arm, dict):
            return [], ("rejected",
                        "checkpoint has no a5_arm descriptor (pre-PR-3 "
                        "checkpoint): experiment-arm identity unverifiable")
        mismatches = []
        if str(arm.get("a2_arm")) != str(cfg.get("a2_arm", "anchor")):
            mismatches.append(
                f"a2_arm ckpt={arm.get('a2_arm')} args={cfg.get('a2_arm')}")
        if float(arm.get("proto_lambda", -1)) != float(
                cfg.get("a5_proto_lambda", 0.5)):
            mismatches.append(
                f"proto_lambda ckpt={arm.get('proto_lambda')} "
                f"args={cfg.get('a5_proto_lambda')}")
        if float(arm.get("proto_tau", -1)) != float(
                cfg.get("a5_proto_tau", 0.07)):
            mismatches.append(
                f"proto_tau ckpt={arm.get('proto_tau')} "
                f"args={cfg.get('a5_proto_tau')}")
        if str(arm.get("u_agg", "")) != str(cfg.get("a5_u_agg", "fedavg")):
            mismatches.append(
                f"u_agg ckpt={arm.get('u_agg')} args={cfg.get('a5_u_agg')}")
        if int(arm.get("rank", -1)) != int(cfg.get("a5_rank", 8)):
            mismatches.append(
                f"rank ckpt={arm.get('rank')} args={cfg.get('a5_rank')}")
        if mismatches:
            return [], ("rejected",
                        "a5_arm descriptor disagrees with args.json: "
                        + "; ".join(mismatches))

    # ---- metrics ---------------------------------------------------------
    rows = []
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            rows.append({
                "after_task": int(rec["after_task"]),
                "avg_accuracy": float(rec["avg_accuracy"]),
                "forgetting": (float(rec["forgetting"])
                               if rec["forgetting"] else None),
                "bwt": float(rec["bwt"]) if rec["bwt"] else None,
            })
    if not rows:
        return [], ("rejected", "summary_metrics.csv is empty")
    return rows, ("accepted", None)


def main():
    p = argparse.ArgumentParser("formal results aggregation (PR-3 guards)")
    p.add_argument("--roots", nargs="+", required=True,
                   help="output roots: {root}/{dataset}/{method}/{run_name}"
                        "/seed_{seed}/")
    p.add_argument("--out", type=Path, default=Path("reports/formal_results.csv"))
    p.add_argument("--report", type=Path,
                   default=Path("reports/formal_report.json"))
    args = p.parse_args()

    long_rows = []      # dataset / method / run_name / seed / after_task
    report = {"runs": [], "n_accepted": 0, "n_rejected": 0}

    for root in args.roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for method_dir in sorted(dataset_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                for run_dir in sorted(method_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    for seed_dir in sorted(run_dir.iterdir()):
                        if not (seed_dir.is_dir()
                                and seed_dir.name.startswith("seed_")):
                            continue
                        try:
                            seed = int(seed_dir.name.split("_", 1)[1])
                        except ValueError:
                            continue
                        rows, (status, reason) = scan_run(seed_dir)
                        entry = {
                            "dataset": dataset_dir.name,
                            "method": method_dir.name,
                            "run_name": run_dir.name,
                            "seed": seed,
                            "status": status,
                        }
                        if reason:
                            entry["reason"] = reason
                        report["runs"].append(entry)
                        if status == "accepted":
                            report["n_accepted"] += 1
                            for r in rows:
                                long_rows.append({
                                    "dataset": dataset_dir.name,
                                    "method": method_dir.name,
                                    "run_name": run_dir.name,
                                    "seed": seed,
                                    "after_task": r["after_task"],
                                    "avg_accuracy": r["avg_accuracy"],
                                    "forgetting": r["forgetting"],
                                    "bwt": r["bwt"],
                                })
                        else:
                            report["n_rejected"] += 1
                            print(f"[REJECT] {dataset_dir.name}/"
                                  f"{method_dir.name}/{run_dir.name}/"
                                  f"seed_{seed}: {reason}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "run_name", "seed", "after_task",
            "avg_accuracy", "forgetting", "bwt"])
        w.writeheader()
        w.writerows(long_rows)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # seed-mean summary per (dataset, method, run_name, after_task)
    grouped = {}
    for r in long_rows:
        grouped.setdefault(
            (r["dataset"], r["method"], r["run_name"], r["after_task"]),
            []).append(r["avg_accuracy"])
    print(f"\naccepted runs: {report['n_accepted']}, "
          f"rejected: {report['n_rejected']}")
    print(f"long table -> {args.out}  ({len(long_rows)} rows)")
    print(f"report     -> {args.report}\n")
    print(f"{'dataset':<12} {'method':<12} {'run':<28} {'task':>4} "
          f"{'AA(seed-mean)':>13} {'n_seeds':>7}")
    for (ds, m, rn, t) in sorted(grouped):
        vals = grouped[(ds, m, rn, t)]
        print(f"{ds:<12} {m:<12} {rn:<28} {t:>4} "
              f"{sum(vals) / len(vals):>13.4f} {len(vals):>7}")


if __name__ == "__main__":
    main()
