"""Fusion erosion probe (RUNCOMMANDS Sec.26, pre-registered 2026-11-02).

Question (decides candidates A/B BEFORE any mechanism code is written):
  A  how much old-task accuracy does a NON-ANCHOR client lose when the
     broadcast replaces its just-trained local prompt pool with the fused
     pool?   (A: row-respecting fusion ceiling)
  B  how much old-task accuracy does the fusion step itself (10-epoch
     current-task-only distillation + Adam wd) erode relative to its own
     base (= the anchor's local pool)?   (B: trajectory-aware fusion)

Design: resume an M2' full run for ONE fusion round with
--fusion_probe_dump, which snapshots every chosen client's post-local
pool (pre-fusion) and the fused pool (post-fusion). This script then
bootstraps the probe run's round checkpoint (post-broadcast: clients hold
the fused pool; heads / protos / model bodies / test splits restored) and
does a PAIRED evaluation: for each (client, task) cell a fixed RNG state
is re-seeded before every pool variant, so shuffle order and dropout
noise are common-mode across variants and the deltas isolate the prompt
pool alone.

  dA(c,t) = acc(own_local)  - acc(fused)   for non-anchor c (A ceiling)
  dB(c,t) = acc(anchor_local) - acc(fused) for all c        (B ceiling)

Sanity gate (fail-closed): the dumped fused pool must be BIT-IDENTICAL to
every bootstrap client's restored prompts (the broadcast did exactly
`self.prompts = prompt`); otherwise the probe is invalid.

Deterministic: read-only probe on checkpoints; no training, no parameter
writes; RNG restored per variant.

Usage (see RUNCOMMANDS Sec.26.2):
  python diagnostics/probe_fusion_erosion.py \
    --run_dir output/ImageNet-R/fedta/inr_fusion_probe_r20/seed_42 \
    --checkpoint round_0020.pth \
    --dump_dir diagnostics_output/fusion_probe/inr \
    --out_csv diagnostics_output/fusion_probe/inr_erosion.csv \
    --summary_json diagnostics_output/fusion_probe/inr_erosion.json \
    --device cuda
"""
import argparse
import csv
import gc
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.diag_utils import bootstrap_server  # noqa: E402
from runtime_utils import rng_state_dict, restore_rng_state  # noqa: E402

# Pre-registered gates (RUNCOMMANDS Sec.26.3; decided BEFORE the data)
GATE_GREEN = 1.0   # pp, mean delta -> GREEN (fund the mechanism)
GATE_RED = 0.3     # pp, mean delta below -> RED (kill the direction)
GATE_POSITIVE_SHARE = 0.6  # cells with delta > 0


def cell_rng_state(client_id, task):
    """Fixed per-(client, task) RNG state re-seeded before EVERY variant so
    shuffle order and any train-mode dropout are common-mode across pool
    variants (paired comparison; probe is read-only so seeding is safe)."""
    seed = int.from_bytes(
        hashlib.sha256(f"fusion_erosion|{client_id}|{task}".encode())
        .digest()[:4], "little")
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    random.seed(seed)
    return rng_state_dict()


def state_dicts_equal(a, b):
    if a is None or b is None or set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def verdict(mean_delta, positive_share):
    if mean_delta is None:
        return "NO_CELLS"
    if mean_delta >= GATE_GREEN:
        v = "GREEN"
    elif mean_delta >= GATE_RED:
        v = "YELLOW"
    else:
        v = "RED"
    if v != "RED" and positive_share < GATE_POSITIVE_SHARE:
        v = "YELLOW" if v == "GREEN" else "RED"
    return v


def main():
    p = argparse.ArgumentParser("fusion erosion probe (Sec.26)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="probe run's round checkpoint (post-broadcast), "
                        "e.g. round_0020.pth")
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--summary_json", default=None)
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args_ns, server = bootstrap_server(
        args.run_dir, args.checkpoint,
        data_path=args.data_path, device=args.device)
    completed_round = int(server.start_round) - 1
    nb_classes = args_ns.nb_classes
    server_task = int(server.task_id)

    dump_dir = Path(args.dump_dir)
    local_pt = dump_dir / f"round_{completed_round:04d}_local.pt"
    fused_pt = dump_dir / f"round_{completed_round:04d}_fused.pt"
    for f in (local_pt, fused_pt):
        if not f.is_file():
            raise SystemExit(f"[probe] dump not found: {f}")
    local = torch.load(local_pt, map_location="cpu")
    fused = torch.load(fused_pt, map_location="cpu")
    if local["round"] != completed_round or fused["round"] != completed_round:
        raise SystemExit(
            f"[probe] dump round mismatch: local={local['round']} "
            f"fused={fused['round']} checkpoint round={completed_round}")
    anchor = int(local["anchor"])
    if int(fused["anchor"]) != anchor:
        raise SystemExit("[probe] anchor mismatch between dumps")
    print(f"[probe] round {completed_round}, anchor=client {anchor}, "
          f"server_task={server_task}")

    # ---- sanity gate: fused dump == bootstrapped (post-broadcast) prompts
    fused_sd = {k: v.to(args_ns.device) for k, v in fused["fused_prompts"].items()}
    for c in server.clients:
        if not state_dicts_equal(c.prompts.state_dict(), fused_sd):
            raise SystemExit(
                f"[probe] SANITY FAIL: client {c.id} restored prompts != "
                "dumped fused pool (broadcast semantics violated) -> "
                "probe invalid, rerun")
    print("[probe] sanity OK: fused dump == checkpoint client prompts")

    # old tasks = tasks strictly before the current one (heads[t] must
    # exist; at a task-first round heads[current] is absent -> reference
    # cells are old-task only, as pre-registered in Sec.26.4-3)
    old_tasks = [t for t in range(server_task)]
    eval_tasks = []
    for t in range(server_task + 1):
        usable = all(
            t <= c.task_id and t < len(c.heads) and c.heads[t] is not None
            and t < len(c.test_loader) for c in server.clients)
        if usable:
            eval_tasks.append(t)
    print(f"[probe] eval tasks: {eval_tasks} (old = {old_tasks})")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "client", "test_task", "variant", "accuracy"])
        for c in server.clients:
            template = deepcopy(c.prompts)
            # dump keys are the chosen client ids (ints, see Server_DF
            # _fusion_probe_dump)
            anchor_sd = local["client_prompts"][anchor]
            variants = {"fused": fused_sd, "anchor_local": anchor_sd}
            variants["own_local"] = (local["client_prompts"][c.id]
                                     if c.id != anchor else anchor_sd)
            for t in eval_tasks:
                cell_rng = cell_rng_state(c.id, t)
                for vname, sd in variants.items():
                    sd = {k: v.to(args_ns.device) for k, v in sd.items()}
                    restore_rng_state(cell_rng)
                    module = deepcopy(template)
                    module.load_state_dict(sd)
                    c.prompts = module
                    c.vit.load_prompts(module)
                    acc = c.evaluate(t, nb_classes)
                    print(f"[probe] client {c.id} task {t} {vname}: {acc:.2f}")
                    w.writerow([completed_round, c.id, t, vname, acc])
            # leave the client's own restored state in place
            c.prompts = template
            c.vit.load_prompts(template)

    # ---- pre-registered quantities (Sec.26.3) -------------------------
    # re-read the CSV (single source of truth)
    cells = {}
    with open(out_path) as f:
        for r in csv.DictReader(f):
            cells[(int(r["client"]), int(r["test_task"]), r["variant"])] = \
                float(r["accuracy"])

    dA = [cells[(c.id, t, "own_local")] - cells[(c.id, t, "fused")]
          for c in server.clients if c.id != anchor for t in old_tasks
          if (c.id, t, "own_local") in cells]
    dB = [cells[(c.id, t, "anchor_local")] - cells[(c.id, t, "fused")]
          for c in server.clients for t in old_tasks
          if (c.id, t, "anchor_local") in cells]

    def _stats(ds):
        if not ds:
            return {"n": 0, "mean": None, "positive_share": None}
        return {"n": len(ds), "mean": round(sum(ds) / len(ds), 4),
                "positive_share": round(
                    sum(1 for d in ds if d > 0) / len(ds), 4)}

    sA, sB = _stats(dA), _stats(dB)
    summary = {
        "run": str(Path(args.run_dir)), "round": completed_round,
        "anchor": anchor, "server_task": server_task,
        "gates": {"green": GATE_GREEN, "red": GATE_RED,
                  "positive_share": GATE_POSITIVE_SHARE},
        "A_discard_delta": sA, "A_verdict": verdict(sA["mean"], sA["positive_share"]),
        "B_erosion_delta": sB, "B_verdict": verdict(sB["mean"], sB["positive_share"]),
        "per_task_A": {
            str(t): round(sum(
                cells[(c.id, t, "own_local")] - cells[(c.id, t, "fused")]
                for c in server.clients if c.id != anchor
                and (c.id, t, "own_local") in cells) / max(
                sum(1 for c in server.clients if c.id != anchor
                    and (c.id, t, "own_local") in cells), 1), 4)
            for t in old_tasks},
        "per_task_B": {
            str(t): round(sum(
                cells[(c.id, t, "anchor_local")] - cells[(c.id, t, "fused")]
                for c in server.clients
                if (c.id, t, "anchor_local") in cells) / max(
                sum(1 for c in server.clients
                    if (c.id, t, "anchor_local") in cells), 1), 4)
            for t in old_tasks},
    }
    print("\n[probe] ===== pre-registered verdicts (Sec.26.3) =====")
    print(f"  A (broadcast discard, non-anchor, old tasks): "
          f"mean {sA['mean']}pp over {sA['n']} cells, "
          f"positive {sA['positive_share']} -> {summary['A_verdict']}")
    print(f"  B (fusion erosion, all clients, old tasks):    "
          f"mean {sB['mean']}pp over {sB['n']} cells, "
          f"positive {sB['positive_share']} -> {summary['B_verdict']}")
    print(f"  per-task A: {summary['per_task_A']}")
    print(f"  per-task B: {summary['per_task_B']}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        print(f"\n[probe] summary -> {args.summary_json}")
    print(f"[probe] cells -> {out_path}")

    del server
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
