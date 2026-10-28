"""Track E: read-only corrected-eval re-scoring of every checkpoint
(2026-10-28, reviewer-approved).

For every task_XX_end.pth checkpoint and every (client, seen task k) it
reports TWO accuracy fields side by side:

  official_compat_acc : verbatim Client_DF.evaluate() (module train flags
                        untouched = official protocol; stochastic because
                        inference-time Dropout stays active)
  corrected_eval_acc  : same forward semantics (incl. the official vit
                        train=True kwarg and batch-8 shuffling), but model
                        + vit + original_model switched to .eval() =
                        Dropout disabled, the corrected deterministic
                        inference protocol

NEVER overwrites anything inside run_dir (history JSON / checkpoints /
old metrics untouched); outputs go to diagnostics_output/ only. Later
writing can then say BOTH "official-compatible, for comparability with
the original FedTA implementation" and "corrected deterministic
evaluation" per the reviewer ruling. No claim is pre-made about the size
of the correction for tasks other than T0 -- it is measured, not assumed.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.probe_head_retrain import bootstrap_with_new_arg_defaults
from diagnostics.probe_eval_dropout_causal import (eval_in_mode,
                                                   head_flags_during_forward)


def discover_checkpoints(run_dir):
    """All task_XX_end.pth in run_dir/checkpoints, ordered by task id."""
    ck_dir = Path(run_dir) / "checkpoints"
    def task_id(path):
        m = re.match(r"task_(\d+)_end\.pth$", path.name)
        return int(m.group(1)) if m else -1
    cands = [p for p in ck_dir.glob("task_*_end.pth") if task_id(p) >= 0]
    return [(task_id(p), "checkpoints/%s" % p.name)
            for p in sorted(cands, key=task_id)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True,
                   help="output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42")
    p.add_argument("--checkpoints", default="",
                   help="comma-separated list overriding auto-discovery "
                        "(relative to run_dir or absolute); default = all "
                        "task_XX_end.pth)")
    p.add_argument("--native_repeats", type=int, default=1,
                   help="repeats per official_compat evaluation (1 = same "
                        "single-draw protocol as the recorded matrix)")
    p.add_argument("--device", default=None)
    p.add_argument("--out_csv",
                   default="diagnostics_output/eval_mode_reeval.csv")
    p.add_argument("--out_json",
                   default="diagnostics_output/eval_mode_reeval.json")
    args = p.parse_args()

    if args.checkpoints:
        ckpts = []
        for item in args.checkpoints.split(","):
            item = item.strip()
            m = re.match(r"task_(\d+)_end\.pth$", Path(item).name)
            ckpts.append((int(m.group(1)) if m else -1, item))
        ckpts.sort(key=lambda t: t[0])
    else:
        ckpts = discover_checkpoints(args.run_dir)
    if not ckpts:
        raise SystemExit("no task_XX_end.pth checkpoints found in %s"
                         % args.run_dir)
    print("re-scoring %d checkpoints: %s"
          % (len(ckpts), [c[1] for c in ckpts]))

    detail_rows = []
    head_checks = {}
    matrices = {"official_compat": {}, "corrected_eval": {}}
    for t, ckpt in ckpts:
        aargs, server = bootstrap_with_new_arg_defaults(
            args.run_dir, ckpt, device=args.device)
        nb_classes = aargs.nb_classes
        clients = server.clients
        for c in clients:
            if c.task_id != t:
                print("WARNING: client %d task_id=%s != checkpoint task %d"
                      "-- evaluate() uses task_id=%s in the vit forward "
                      "(official semantics preserved as-is)"
                      % (c.id, c.task_id, t, c.task_id))
        corrected_head_failures = 0
        for k in range(t + 1):
            off_accs, corr_accs = [], []
            for c in clients:
                off = float(sum(eval_in_mode(c, k, nb_classes, "native")
                               for _ in range(args.native_repeats))
                           / args.native_repeats)
                corr, head_flags = head_flags_during_forward(
                    c, k, nb_classes, "full_eval")
                # fail-closed runtime check: the head actually installed
                # by load_head (which bypasses model.eval()) must be in
                # eval mode AT FORWARD TIME under the corrected protocol
                if head_flags != [False]:
                    corrected_head_failures += 1
                off_accs.append(off)
                corr_accs.append(corr)
                detail_rows.append([ckpt, t, c.id, k, round(off, 4),
                                    round(corr, 4)])
            matrices["official_compat"].setdefault(t, {})[k] = \
                round(sum(off_accs) / len(off_accs), 4)
            matrices["corrected_eval"].setdefault(t, {})[k] = \
                round(sum(corr_accs) / len(corr_accs), 4)
            print("%s task %d <- %d: official %.4f | corrected %.4f "
                  "(delta %+.4fpp)"
                  % (ckpt, t, k, matrices["official_compat"][t][k],
                     matrices["corrected_eval"][t][k],
                     matrices["corrected_eval"][t][k]
                     - matrices["official_compat"][t][k]))
        if corrected_head_failures:
            print("WARNING: %s: %d corrected evaluations ran with a "
                  "train-mode loaded head (load_head bypass not covered)"
                  % (ckpt, corrected_head_failures))
        head_checks[ckpt] = corrected_head_failures
        del server
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aa = {}
    for proto, mat in matrices.items():
        aa[proto] = {t: round(sum(mat[t][k] for k in mat[t]) / len(mat[t]),
                              4) for t in mat}

    summary = {
        "probe": "eval_mode_reeval",
        "track": "E_evaluation_correctness",
        "run_dir": args.run_dir,
        "protocol": {
            "official_compat": "verbatim evaluate(); module train flags "
                               "untouched (inference-time Dropout ACTIVE, "
                               "stochastic)",
            "corrected_eval": "same forward semantics; model+vit+"
                              "original_model .eval() (Dropout disabled, "
                              "deterministic)",
            "native_repeats": args.native_repeats,
            "aggregation": "unweighted mean over clients (matrix semantics)",
            "read_only": "outputs in diagnostics_output/ only; run_dir "
                         "history never modified",
        },
        "matrices": {proto: {str(t): {str(k): v for k, v in row.items()}
                             for t, row in mat.items()}
                     for proto, mat in matrices.items()},
        "average_accuracy": aa,
        "corrected_head_train_mode_failures": head_checks,
        "per_client_detail": [dict(zip(
            ("checkpoint", "task_t", "client", "eval_task",
             "official_compat_acc", "corrected_eval_acc"), r))
            for r in detail_rows],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "task_t", "client", "eval_task",
                    "official_compat_acc", "corrected_eval_acc"])
        w.writerows(detail_rows)
        w.writerow([])
        w.writerow(["checkpoint", "task_t", "row_type", "eval_task",
                    "official_compat_acc", "corrected_eval_acc"])
        for t, ckpt in ckpts:
            for k in range(t + 1):
                w.writerow([ckpt, t, "AGG_MEAN_OVER_CLIENTS", k,
                            matrices["official_compat"][t][k],
                            matrices["corrected_eval"][t][k]])
            w.writerow([ckpt, t, "AA", "mean(k=0..%d)" % t,
                        aa["official_compat"][t], aa["corrected_eval"][t]])
    print("\n=== Track E re-scoring (official_compat -> corrected_eval) ===")
    for t, ckpt in ckpts:
        print("%s: AA %s -> %s" % (ckpt, aa["official_compat"][t],
                                   aa["corrected_eval"][t]))
    print("wrote %s and %s" % (out_json, args.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
