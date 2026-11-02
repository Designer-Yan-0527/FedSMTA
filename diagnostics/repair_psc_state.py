"""One-off checkpoint repair for the dead-code build_checkpoint bug.

Bug (fixed in Models/Server_DF.build_checkpoint): the psc_state block was
inserted after the unconditional `return {...}` literal, so EVERY
checkpoint saved by the --psc short-run arms lacks the server-side
psc_state key. The fail-closed loader (correctly) refuses to resume or
bootstrap probes from such checkpoints.

The server PSC state of these two runs is provably the EMPTY triple
    {core_slots: [], pair_streak: {}, merged_pairs: []}
from their own train.log evidence:
  * both task-boundary fusion rounds printed "0 merge candidates" ->
    no merge was ever committed -> core_slots and merged_pairs stayed [];
  * every fusion round printed "0 active streaks" -> pair_streak stayed {}.
This script VERIFIES that log evidence before repairing (fail-closed):
if any "merged pair" line or non-zero streak is found, it refuses.

Repair is atomic per file (save to *.tmp, os.replace) and one file at a
time (memory-friendly). Only checkpoints whose stored args have psc=True
are touched; anything else aborts. A verification pass reloads every file
afterwards.
"""
import gc
import os
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIRS = [
    "output/cifar100/fedta/cifar_psc_M2p_psc_t1/seed_42",
    "output/ImageNet-R/fedta/inr_psc_M2p_p40_psc_t1/seed_42",
]

EMPTY_STATE = {"core_slots": [], "pair_streak": {}, "merged_pairs": []}


def verify_log_evidence(run_dir: Path) -> None:
    """Refuse to repair unless the run's own log proves the empty state."""
    log = run_dir / "logs" / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "[PSC]" in text, f"{run_dir}: no [PSC] lines -- not a PSC run?"
    assert "merged pair" not in text, (
        f"{run_dir}: log shows committed merges -- the empty-triple repair "
        "would be WRONG; aborting (rerun from scratch instead)")
    streaks = re.findall(r"(\d+) active streaks", text)
    bad = [s for s in streaks if s != "0"]
    assert not bad, (
        f"{run_dir}: log shows non-zero streaks {bad} -- repair would be "
        "WRONG; aborting")
    boundaries = re.findall(r"\(task-boundary fusion\): (\d+) merge candidates",
                            text)
    assert boundaries and all(b == "0" for b in boundaries), (
        f"{run_dir}: boundary rounds not all zero candidates: {boundaries}")
    print(f"  log evidence OK: {len(streaks)} rounds all 0 streaks, "
          f"boundaries {boundaries} -> empty triple is provably correct")


def repair_run(run_dir: Path) -> int:
    print(f"== {run_dir}")
    verify_log_evidence(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    files = sorted(ckpt_dir.glob("*.pth"))
    assert files, f"{run_dir}: no checkpoints found"
    n_repaired = 0
    for path in files:
        ckpt = torch.load(str(path), map_location="cpu")
        args = ckpt.get("args") or {}
        assert args.get("psc") is True, (
            f"{path}: stored args do not have psc=True -- refusing to touch")
        if "psc_state" in ckpt:
            print(f"  {path.name}: already has psc_state (skip)")
            del ckpt
            gc.collect()
            continue
        ckpt["psc_state"] = dict(EMPTY_STATE)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, str(tmp))
        os.replace(str(tmp), str(path))
        n_repaired += 1
        print(f"  {path.name}: psc_state injected "
              f"(round {ckpt.get('completed_round')})")
        del ckpt
        gc.collect()
    # verification pass
    for path in files:
        ckpt = torch.load(str(path), map_location="cpu")
        assert "psc_state" in ckpt, f"{path.name}: still missing after repair"
        del ckpt
        gc.collect()
    print(f"  repaired {n_repaired}/{len(files)} files, all verified")
    return n_repaired


def main():
    run_dirs = [REPO_ROOT / d for d in (
        sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_RUN_DIRS)]
    for d in run_dirs:
        assert d.is_dir(), f"{d}: run dir not found"
    total = sum(repair_run(d) for d in run_dirs)
    print(f"\nDONE: {total} checkpoints repaired")


if __name__ == "__main__":
    main()
