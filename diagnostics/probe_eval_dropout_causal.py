"""Track E (Evaluation Correctness): Dropout-at-eval causal toggle probe
(read-only, 2026-10-28).

Finding under review: Client_DF.evaluate() never switches the Chead
(Linear 1536->512 -> ReLU -> Dropout(0.5) -> 200) out of train mode, so
Dropout stays ACTIVE at inference -- stochastic logits and a systematic
accuracy under-estimate (T0: matrix 82.87 vs 84.81 measured in eval mode,
+1.94pp). Before the full corrected-eval re-scoring of every checkpoint,
this probe closes the causal loop ON THE SAME checkpoint AND THE SAME
test loader, with three toggle states (reviewer-approved protocol):

  native    : verbatim official evaluate() -- module train flags untouched
  head_eval : ONLY client.model (Tail_Anchor incl. Chead) switched to
              .eval(); Tail_Anchor has no BatchNorm and no Dropout outside
              the Chead, so this toggles exactly the Chead Dropout(0.5)
  full_eval : model + vit + original_model all .eval() = the corrected
              deterministic-inference protocol (forward semantics -- incl.
              the official `train=True` vit kwarg and batch-8 shuffling --
              are kept verbatim; only module-level Dropout is disabled)

Expected if the Dropout explanation is causal:
  native ~ 82.87 with stochastic scatter across repeats (Dropout sampling)
  head_eval ~ full_eval ~ 84.81 with near-zero scatter across repeats

Read-only: bootstraps via diagnostics.probe_head_retrain (args.json +
checkpoint restore; never writes into run_dir); outputs go to
diagnostics_output/ only. Module train/eval flags are always restored
after each call.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.probe_head_retrain import bootstrap_with_new_arg_defaults

TOGGLE_MODULES = ("model", "vit", "original_model")


def snapshot_flags(client):
    """Module training flags of every toggled module, for exact restore."""
    return {name: bool(getattr(client, name).training)
            for name in TOGGLE_MODULES}


def restore_flags(client, flags):
    for name, flag in flags.items():
        getattr(client, name).train(flag)


def eval_in_mode(client, task, nb_classes, mode):
    """One official client.evaluate() call under a toggle state.

    The official path is called VERBATIM (batch 8, shuffle=True, load_head,
    vit train=True kwarg, class_mask); only the module-level train/eval
    flags differ per mode, and they are restored in a finally block."""
    flags = snapshot_flags(client)
    try:
        if mode == "native":
            pass
        elif mode == "head_eval":
            # Tail_Anchor: no BatchNorm, no Dropout outside Chead ->
            # this isolates exactly the Chead Dropout(0.5).
            client.model.eval()
        elif mode == "full_eval":
            client.model.eval()
            client.vit.eval()
            client.original_model.eval()
        else:
            raise ValueError("unknown mode: %r" % mode)
        return client.evaluate(task, nb_classes)
    finally:
        restore_flags(client, flags)


def dropout_inventory(client):
    """Static evidence: which modules carry Dropout / BatchNorm."""
    inv = {}
    for name in TOGGLE_MODULES:
        mod = getattr(client, name)
        drops = [m for m in mod.modules() if isinstance(m, torch.nn.Dropout)]
        bns = [m for m in mod.modules()
               if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                 torch.nn.BatchNorm3d))]
        inv[name] = {
            "training_flag": bool(mod.training),
            "n_dropout": len(drops),
            "dropout_p": sorted({float(m.p) for m in drops}),
            "n_batchnorm": len(bns),
        }
    return inv


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True,
                   help="output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42")
    p.add_argument("--checkpoint", default="checkpoints/task_00_end.pth")
    p.add_argument("--task", type=int, default=0,
                   help="evaluated task index (heads[task] / test_loader[task])")
    p.add_argument("--native_repeats", type=int, default=15,
                   help="native-mode repeats (stochastic scatter evidence)")
    p.add_argument("--eval_repeats", type=int, default=3,
                   help="head_eval / full_eval repeats (determinism evidence)")
    p.add_argument("--native_t0", type=float, default=82.87,
                   help="matrix reference for the native protocol")
    p.add_argument("--eval_mode_ref", type=float, default=84.81,
                   help="H1-probe eval-mode calibration reference")
    p.add_argument("--device", default=None)
    p.add_argument("--out_json",
                   default="diagnostics_output/eval_dropout_causal.json")
    p.add_argument("--out_csv",
                   default="diagnostics_output/eval_dropout_causal.csv")
    args = p.parse_args()

    aargs, server = bootstrap_with_new_arg_defaults(
        args.run_dir, args.checkpoint, device=args.device)
    nb_classes = aargs.nb_classes
    clients = server.clients
    print("bootstrapped %s | %d clients | task %d | native x%d, eval x%d"
          % (args.run_dir, len(clients), args.task,
             args.native_repeats, args.eval_repeats))

    modes = (("native", args.native_repeats),
             ("head_eval", args.eval_repeats),
             ("full_eval", args.eval_repeats))
    per_client = {}
    inventory = {}
    rows = []
    for c in clients:
        inventory[c.id] = dropout_inventory(c)
        rec = {}
        for mode, n in modes:
            accs = [eval_in_mode(c, args.task, nb_classes, mode)
                    for _ in range(n)]
            rec[mode] = accs
            for i, a in enumerate(accs):
                rows.append([c.id, mode, i, round(a, 4)])
        per_client[c.id] = rec
        nat = rec["native"]
        print("client %d: native %.2f +- %.2f (n=%d, min %.2f max %.2f)"
              " | head_eval %.2f | full_eval %.2f"
              % (c.id, float(np.mean(nat)), float(np.std(nat)), len(nat),
                 float(np.min(nat)), float(np.max(nat)),
                 float(np.mean(rec["head_eval"])),
                 float(np.mean(rec["full_eval"]))))

    agg = {}
    for mode, _ in modes:
        means = [float(np.mean(per_client[c.id][mode])) for c in clients]
        agg[mode] = {
            "mean": round(float(np.mean(means)), 4),
            "std_over_clients": round(float(np.std(means)), 4),
            "min": round(float(np.min(means)), 4),
            "max": round(float(np.max(means)), 4),
            "per_client_mean": [round(m, 4) for m in means],
        }

    native_std = [float(np.std(per_client[c.id]["native"])) for c in clients]
    native_spread = [float(np.max(l) - np.min(l))
                     for c in clients
                     for l in (per_client[c.id]["native"],)]
    eval_spread = 0.0
    for c in clients:
        for mode in ("head_eval", "full_eval"):
            l = per_client[c.id][mode]
            eval_spread = max(eval_spread, float(np.max(l) - np.min(l)))

    dropout_effect = round(agg["full_eval"]["mean"] - agg["native"]["mean"], 4)
    head_vs_full = round(agg["head_eval"]["mean"] - agg["full_eval"]["mean"],
                         4)
    native_stochastic = float(np.mean(native_std)) >= 0.05
    eval_stable = eval_spread <= 0.5 and eval_spread <= 2.0 * float(
        np.mean(native_spread))
    causal_confirmed = bool(dropout_effect >= 1.0
                            and abs(head_vs_full) <= 0.5
                            and native_stochastic
                            and eval_stable)

    summary = {
        "probe": "eval_dropout_causal_toggle",
        "track": "E_evaluation_correctness",
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "evaluated_task": args.task,
        "protocol": {
            "native": "verbatim evaluate(), module train flags untouched",
            "head_eval": "only Tail_Anchor (incl. Chead) .eval(); no other "
                         "change -- isolates the Chead Dropout(0.5)",
            "full_eval": "model+vit+original_model .eval(); forward "
                         "semantics (vit train=True kwarg, batch-8 shuffle) "
                         "kept verbatim = corrected deterministic protocol",
            "native_repeats": args.native_repeats,
            "eval_repeats": args.eval_repeats,
            "aggregation": "unweighted mean over clients (matrix semantics)",
            "read_only": True,
        },
        "references": {"matrix_native": args.native_t0,
                       "eval_mode_calibration": args.eval_mode_ref},
        "module_inventory": inventory,
        "aggregate": agg,
        "native_repeat_std_per_client": [round(s, 4) for s in native_std],
        "native_repeat_spread_max": round(float(np.max(native_spread)), 4),
        "eval_repeat_spread_max": round(eval_spread, 4),
        "dropout_effect_pp": dropout_effect,
        "head_eval_minus_full_eval_pp": head_vs_full,
        "verdict": "CAUSAL_CONFIRMED" if causal_confirmed else "INCONCLUSIVE",
        "verdict_criteria": {
            "dropout_effect >= 1.0": dropout_effect >= 1.0,
            "|head_eval - full_eval| <= 0.5": abs(head_vs_full) <= 0.5,
            "native stochastic (mean repeat std >= 0.05)": native_stochastic,
            "eval repeats stable": eval_stable,
        },
        "per_client_accs": {str(cid): {m: [round(a, 4) for a in l]
                                       for m, l in rec.items()}
                             for cid, rec in per_client.items()},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "mode", "repeat_idx", "acc"])
        w.writerows(rows)
    print("\n=== Track E causal toggle (task %d) ===" % args.task)
    print("native    %.4f  [matrix %.2f]  repeat std mean %.3f, spread max"
          " %.2f" % (agg["native"]["mean"], args.native_t0,
                     float(np.mean(native_std)),
                     float(np.max(native_spread))))
    print("head_eval %.4f  (Chead Dropout isolated)" % agg["head_eval"]["mean"])
    print("full_eval %.4f  [eval-mode ref %.2f]  repeat spread max %.2f"
          % (agg["full_eval"]["mean"], args.eval_mode_ref, eval_spread))
    print("dropout effect %+.4fpp | head_eval - full_eval %+.4fpp"
          % (dropout_effect, head_vs_full))
    print("VERDICT: %s" % summary["verdict"])
    print("wrote %s and %s" % (out_json, args.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
