"""Unified metrics recorder for the external-baseline protocol adapters.

Emits the SAME record schema and CSV layout as the main repo's RunManager
(runtime_utils.py), so every method -- fedta/A2/A3 (main repo, RunManager)
and the external baselines (FedMGP, C2Prompt, ...) -- lands in one output
tree and feeds one three-level (dataset / seed / task) aggregation:

    {out_root}/{dataset}/{method}/{run_name}/seed_{seed}/
        meta.json                       run provenance
        metrics/records.jsonl           append-only raw records
        metrics/client_X_accuracy.csv   (client_id, after_task, test_task,
                                        accuracy) -- all records, sorted
        metrics/accuracy_matrix.csv     client-mean R[t][k], "latest wins"
                                        per (after_task, test_task, client),
                                        header after_task,T0..T{n-1}
                                        (identical format to
                                        runtime_utils.RunManager)
        metrics/summary_metrics.csv     per after_task: avg_accuracy /
                                        forgetting / bwt (identical
                                        formulas to RunManager
                                        ._summary_for_matrix)

R[t][k] = client-mean accuracy AFTER task t on test task k.

This module is deliberately torch-free (pure bookkeeping); the adapters
own the evaluation call sites:

  - run_c2p.py OBSERVES C2Prompt's native boundary eval (which already
    evaluates every trained client-task on the manifest test pool), so
    the recorder adds zero extra computation and zero RNG impact;
  - run_fedmgp.py ADDS an RNG-neutral boundary eval over all seen tasks
    (FedMGP natively evaluates only task 0 + the current task), so the
    full R[t][k] matrix exists for every boundary.
"""

import csv
import json
import time
from pathlib import Path


class MetricsRecorder(object):
    """Records (client_id, after_task, test_task, accuracy) tuples and
    rewrites the RunManager-format CSVs on every flush()."""

    def __init__(self, out_root, dataset, method, seed,
                 run_name="manifest_adapter", meta=None):
        self.dataset = str(dataset)
        self.method = str(method)
        self.seed = int(seed)
        self.run_name = str(run_name)
        self.run_dir = (Path(out_root) / self.dataset / self.method
                        / self.run_name / f"seed_{self.seed}")
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.metrics_dir / "records.jsonl"
        self.records = []

        meta = dict(meta or {})
        meta.update({
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "run_name": self.run_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "schema": "fedsmta-adapter-metrics-v1",
        })
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    # -- recording ---------------------------------------------------------

    def record(self, client_id, after_task, test_task, accuracy,
               phase="native", **extra):
        """Append one R[after_task][test_task] observation for one client.

        phase: 'boundary' (task-boundary full-matrix value; wins in the
        matrix because flush() keeps the LATEST record per key) or
        'native' (a method-native mid-round observation, kept in
        records.jsonl / client CSVs for traceability).
        """
        rec = {
            "client_id": int(client_id),
            "after_task": int(after_task),
            "test_task": int(test_task),
            "accuracy": float(accuracy),
            "phase": str(phase),
        }
        rec.update(extra)
        self.records.append(rec)
        # defensive: the metrics dir may have been removed after init
        # (e.g. an over-eager cleanup); never let a log write kill training
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # -- CSV emission (formats mirror runtime_utils.RunManager) ------------

    def flush(self, task_num):
        """Rewrite client CSVs, the client-mean accuracy matrix and the
        summary metrics from the full record list. Call after every task
        boundary (cheap: a few hundred rows)."""
        task_num = int(task_num)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        by_client = {}
        for r in self.records:
            by_client.setdefault(r["client_id"], []).append(r)

        for cid, rows in by_client.items():
            rows = sorted(rows, key=lambda r: (r["after_task"],
                                               r["test_task"]))
            path = self.metrics_dir / f"client_{cid}_accuracy.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["client_id", "after_task", "test_task",
                            "accuracy"])
                for r in rows:
                    w.writerow([r["client_id"], r["after_task"],
                                r["test_task"], f"{r['accuracy']:.2f}"])

        # latest value per (after_task, test_task, client), then client mean
        latest = {}
        for r in self.records:
            latest[(r["after_task"], r["test_task"],
                    r["client_id"])] = r["accuracy"]
        after_tasks = sorted({key[0] for key in latest})

        with open(self.metrics_dir / "accuracy_matrix.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["after_task"] + [f"T{t}" for t in range(task_num)])
            for a in after_tasks:
                row = [a]
                for t in range(task_num):
                    vals = [v for (aa, tt, _c), v in latest.items()
                            if aa == a and tt == t]
                    row.append(f"{sum(vals) / len(vals):.2f}" if vals
                               else "")
                w.writerow(row)

        # summary metrics from the client-mean matrix R[t][k]
        R = {}
        for a in after_tasks:
            R[a] = {}
            for t in range(task_num):
                vals = [v for (aa, tt, _c), v in latest.items()
                        if aa == a and tt == t]
                if vals:
                    R[a][t] = sum(vals) / len(vals)

        with open(self.metrics_dir / "summary_metrics.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["after_task", "avg_accuracy", "forgetting", "bwt"])
            for a in after_tasks:
                seen = [k for k in range(a + 1) if k in R.get(a, {})]
                if not seen:
                    continue
                avg = sum(R[a][k] for k in seen) / len(seen)
                olds = [k for k in range(a)
                        if k in R.get(a, {}) and k in R.get(k, {})]
                if olds:
                    forg = sum(
                        max(0.0, max(R[l][k] for l in range(k, a)
                                     if k in R.get(l, {})) - R[a][k])
                        for k in olds) / len(olds)
                    bwt = sum(R[a][k] - R[k][k] for k in olds) / len(olds)
                    w.writerow([a, f"{avg:.4f}", f"{forg:.4f}",
                                f"{bwt:.4f}"])
                else:
                    w.writerow([a, f"{avg:.4f}", "", ""])
