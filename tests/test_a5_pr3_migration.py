"""A5 PR-3 fail-closed migration-guard tests (GPU, short runs).

Closes the single OPEN item of the PR-2 external review (CONDITIONAL PASS,
2026-10-21): a PR-1-era fedta_a5 checkpoint (no a5_proto_state / no
a5_proto_lambda/tau in its args) must NOT silently resume into the PR-2
prototype-CE configuration.

Required regressions (reviewer's wording):
  (a) old FedTA/A2 resume is unaffected;
  (b) legacy PR-1 A5 is refused by default;
  (c) a PR-2 A5 checkpoint fully resumes under the same configuration.

Plus the reviewer's "not just lambda/tau" requirement: explicit migration
must still validate the table schema, version and resume boundary.

Cases:
  M0  sanity: PR-2 A5 checkpoints carry a5_proto_state + lambda/tau
      (version == task_id == 0 at the task-0 boundary).
  M1  legacy A5 (state + hparams stripped), no flag  -> REFUSED.
  M2  a5_proto_state stripped only,           no flag  -> REFUSED.
  M3  lambda/tau stripped only,               no flag  -> REFUSED.
  M4  legacy A5 mid-task checkpoint WITH --allow_proto_migration
      -> REFUSED (must resume at a TASK BOUNDARY).
  M5  legacy A5 boundary checkpoint WITH --allow_proto_migration
      -> SUCCEEDS; table rebuilt from task 1 ONLY (keys == task-1 global
      seen set, version == 1); lambda/tau ADOPTED from the CLI.
  M6  PR-2 A5 boundary checkpoint, no flag (regression (c))
      -> SUCCEEDS; final table keys/version/counts == the uninterrupted
      run's final table.
  M7  fedta_a2 run + --resume auto (regression (a)) -> SUCCEEDS;
      --allow_proto_migration with a NON-A5 method -> REFUSED (M7b).
  M8  hybrid legacy (table present, lambda/tau stripped) WITH flag
      -> SUCCEEDS and the table is RESTORED (final keys == uninterrupted
      run); corrupt version is refused both under migration (M8b) and on
      the normal path (M8c).
  M9  PR-3 increment (external review follow-up):
      M9a PR-2 checkpoint resumed WITH --allow_proto_migration is a
          NO-OP: lambda/tau unchanged (from the checkpoint, not the CLI),
          final table bit-identical to the uninterrupted run.
      M9b NaN-poisoned a5_proto_state -> REFUSED (fail-closed).
      M9c every checkpoint written by a MIGRATED run carries the
          a5_proto_migration provenance marker (checked on M5/M8 outputs).
      M9d chained resume from a migrated run's checkpoint INHERITS the
          marker (no flag needed — it is provenance, not permission) and
          the training trajectory is unchanged (table bit-identical).
      M9e PR-2 checkpoint + CLI lambda/tau MISMATCH + flag -> REFUSED by
          RESUME_MUST_MATCH (the flag authorizes legacy migration only;
          it must never bypass config-consistency checks or silently
          override checkpoint values).

Isolation: RunManager infers run_dir from an explicit --resume path
(ckpt.parent.parent), so every synthesized checkpoint lives in its own
scratch directory and never pollutes the source runs.

Usage (from repo root, on the server; Tail_Anchor hardcodes cuda):
  python tests/test_a5_pr3_migration.py --data_path <DATA_PATH>
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_resume_regression import torch_load_compat  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("A5 PR-3 legacy-migration guard tests")
    p.add_argument("--data_name", default="cifar100")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=1, type=int)
    p.add_argument("--work_dir", default="output/regression_a5_pr3",
                   type=Path)
    return p.parse_args()


def run_main(extra, expect_fail=False, expect_msg=None, label=""):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    if expect_fail:
        assert proc.returncode != 0, (
            f"{label}: expected failure but command succeeded")
        combined = (proc.stdout or "") + (proc.stderr or "")
        if expect_msg is not None:
            assert expect_msg in combined, (
                f"{label}: expected error message containing "
                f"{expect_msg!r}, got tail:\n"
                + "\n".join(combined.splitlines()[-30:]))
        print(f"[PASS] {label}: rejected as expected")
    else:
        if proc.returncode != 0:
            print((proc.stdout or "")[-2000:])
            print((proc.stderr or "")[-2000:])
            raise RuntimeError(f"{label}: command failed")
        print(f"[PASS] {label}")
    return proc


def make_legacy(src_ckpt, dst_dir, name, strip_state=True,
                strip_hparams=True, force_version=None):
    """Synthesize a checkpoint copy in its own scratch run-dir. strip_state
    removes a5_proto_state (PR-1 schema); strip_hparams removes
    a5_proto_lambda/tau from the recorded args; force_version corrupts the
    table version."""
    dst = dst_dir / "checkpoints" / f"{name}.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch_load_compat(src_ckpt)
    if strip_state:
        ckpt.pop("a5_proto_state", None)
    if strip_hparams:
        ckpt.get("args", {}).pop("a5_proto_lambda", None)
        ckpt.get("args", {}).pop("a5_proto_tau", None)
    if force_version is not None:
        ckpt["a5_proto_state"]["version"] = force_version
    import torch
    torch.save(ckpt, dst)
    return dst


def task_seen_classes(ckpt, task_id):
    """Global seen-class set of `task_id` from the checkpoint's
    outer_split_manifest (client -> task -> class_mask)."""
    seen = set()
    manifest = ckpt["outer_split_manifest"]
    for client in manifest.values():
        seen.update(int(c) for c in client[str(task_id)]["class_mask"])
    return seen


def main():
    args = parse_args()

    base = ["--data_name", args.data_name, "--data-path", args.data_path,
            "--seed", str(args.seed), "--device", args.device,
            "--global_epoch", str(args.global_epoch),
            "--task_num", str(args.task_num),
            "--local_epoch", str(args.local_epoch),
            "--deterministic", "--save_every", "1",
            "--output_dir", str(args.work_dir)]

    run_a5 = (args.work_dir / args.data_name / "fedta_a5"
              / "mig_a5" / f"seed_{args.seed}")
    run_a2 = (args.work_dir / args.data_name / "fedta_a2"
              / "mig_a2" / f"seed_{args.seed}")
    synth = args.work_dir / "synth"
    for run in (run_a5, run_a2, synth):
        if run.exists():
            print(f"[cleanup] removing previous test dir {run}")
            shutil.rmtree(run)

    ge = args.global_epoch
    total = args.task_num * ge
    assert args.task_num >= 2, "the table-rebuild assertions need >= 2 tasks"

    # ---- source runs --------------------------------------------------
    # Run A: uninterrupted PR-2 A5 (all tasks) — reference tables.
    run_main(base + ["--method", "fedta_a5", "--run_name", "mig_a5",
                     "--max_rounds", str(total)],
             label="A5 uninterrupted run")
    ck_a0 = torch_load_compat(run_a5 / "checkpoints" / "task_00_end.pth")
    ck_a1 = torch_load_compat(run_a5 / "checkpoints" / "task_01_end.pth")

    # M0 sanity: PR-2 schema preconditions of this test itself.
    assert ck_a0["server_task_id"] == 0 and ck_a0["completed_round"] == ge - 1
    assert isinstance(ck_a0.get("a5_proto_state"), dict), \
        "M0: PR-2 A5 checkpoint must carry a5_proto_state"
    assert ck_a0["a5_proto_state"]["version"] == 0, \
        "M0: task-0 boundary checkpoint must carry version == 0"
    assert "a5_proto_lambda" in ck_a0["args"] \
        and "a5_proto_tau" in ck_a0["args"], \
        "M0: PR-2 A5 checkpoint args must carry lambda/tau"
    print("[PASS] M0: PR-2 A5 schema sanity (state + version + hparams)")

    # Run B: fedta_a2 (regression (a)) — 3 rounds then --resume auto.
    run_main(base + ["--method", "fedta_a2", "--a2_arm", "anchor",
                     "--run_name", "mig_a2", "--max_rounds",
                     str(ge + 1)],
             label="A2 run (pre-resume)")
    run_main(base + ["--method", "fedta_a2", "--a2_arm", "anchor",
                     "--run_name", "mig_a2", "--resume", "auto",
                     "--max_rounds", str(total)],
             label="M7: A2 resume unaffected")

    # M7b: the migration flag is A5-only (fail-fast for every other method)
    run_main(base + ["--method", "fedta_a2", "--a2_arm", "anchor",
                     "--run_name", "mig_a2_flag", "--max_rounds", "1",
                     "--allow_proto_migration"],
             expect_fail=True, expect_msg="--allow_proto_migration is ONLY",
             label="M7b: flag rejected for non-A5 method")

    # ---- synthesized legacy checkpoints (each in its own run dir) ------
    boundary = run_a5 / "checkpoints" / "task_00_end.pth"
    midtask = run_a5 / "checkpoints" / f"round_{ge - 2:04d}.pth"
    assert midtask.is_file(), f"mid-task checkpoint missing: {midtask}"

    legacy_full = make_legacy(boundary, synth / "legacy_full", "legacy",
                              strip_state=True, strip_hparams=True)
    legacy_mid = make_legacy(midtask, synth / "legacy_mid", "legacy_mid",
                             strip_state=True, strip_hparams=True)
    no_state = make_legacy(boundary, synth / "no_state", "no_state",
                           strip_state=True, strip_hparams=False)
    no_hparams = make_legacy(boundary, synth / "no_hparams", "no_hparams",
                             strip_state=False, strip_hparams=True)
    hybrid = make_legacy(boundary, synth / "hybrid", "hybrid",
                         strip_state=False, strip_hparams=True)
    badver = make_legacy(boundary, synth / "badver", "badver",
                         strip_state=False, strip_hparams=True,
                         force_version=5)
    badver_normal = make_legacy(boundary, synth / "badver_normal",
                                "badver_normal", strip_state=False,
                                strip_hparams=False, force_version=5)
    pr2_copy = synth / "pr2_resume" / "checkpoints" / "ckpt.pth"
    pr2_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(boundary, pr2_copy)

    a5_resume = ["--method", "fedta_a5"]

    # ---- default rejections (regression (b)) ---------------------------
    run_main(base + a5_resume + ["--resume", str(legacy_full),
                                 "--max_rounds", str(total)],
             expect_fail=True, expect_msg="Legacy A5 checkpoint",
             label="M1: legacy A5 (state+hparams stripped) refused by default")
    run_main(base + a5_resume + ["--resume", str(no_state),
                                 "--max_rounds", str(total)],
             expect_fail=True, expect_msg="Legacy A5 checkpoint",
             label="M2: a5_proto_state stripped only, refused by default")
    run_main(base + a5_resume + ["--resume", str(no_hparams),
                                 "--max_rounds", str(total)],
             expect_fail=True, expect_msg="Legacy A5 checkpoint",
             label="M3: lambda/tau stripped only, refused by default")

    # ---- boundary + schema validation under explicit migration ---------
    run_main(base + a5_resume + ["--resume", str(legacy_mid),
                                 "--max_rounds", str(total),
                                 "--allow_proto_migration"],
             expect_fail=True, expect_msg="TASK BOUNDARY",
             label="M4: mid-task migration refused (boundary required)")
    run_main(base + a5_resume + ["--resume", str(badver),
                                 "--max_rounds", str(total),
                                 "--allow_proto_migration"],
             expect_fail=True, expect_msg="version",
             label="M8b: corrupt table version refused under migration")
    run_main(base + a5_resume + ["--resume", str(badver_normal),
                                 "--max_rounds", str(total)],
             expect_fail=True, expect_msg="version",
             label="M8c: corrupt table version refused on normal path")

    # ---- explicit migration happy path (M5) ----------------------------
    # lambda/tau deliberately differ from the source run (0.25 / 0.11) to
    # prove they are ADOPTED from the CLI, not inherited silently.
    run_main(base + a5_resume + ["--resume", str(legacy_full),
                                 "--max_rounds", str(total),
                                 "--a5_proto_lambda", "0.25",
                                 "--a5_proto_tau", "0.11",
                                 "--allow_proto_migration"],
             label="M5: explicit legacy migration succeeds")
    ck_m5 = torch_load_compat(synth / "legacy_full" / "checkpoints"
                              / "task_01_end.pth")
    m5_state = ck_m5["a5_proto_state"]
    assert m5_state["version"] == 1, \
        f"M5: expected version 1 after task-1 rebuild, got {m5_state['version']}"
    t1_seen = task_seen_classes(ck_a1, 1)
    t0_seen = task_seen_classes(ck_a1, 0)
    m5_keys = set(m5_state["sums"])
    assert m5_keys == t1_seen, (
        f"M5: migrated table keys {len(m5_keys)} != task-1 global seen "
        f"{len(t1_seen)} (the rebuild must cover exactly the post-migration "
        f"task; missing={sorted(t1_seen - m5_keys)[:10]}, "
        f"extra={sorted(m5_keys - t1_seen)[:10]})")
    leaked = m5_keys - t1_seen
    assert not leaked, f"M5: table leaked non-task-1 classes {sorted(leaked)}"
    assert float(ck_m5["args"]["a5_proto_lambda"]) == 0.25 \
        and float(ck_m5["args"]["a5_proto_tau"]) == 0.11, \
        "M5: lambda/tau must be adopted from the current command line"
    print("[PASS] M5: table rebuilt from task 1 only "
          f"({len(m5_keys)} classes, version 1), lambda/tau adopted from CLI; "
          f"task-0-only classes absent "
          f"({len(t0_seen - t1_seen)} excluded by design)")

    # ---- PR-2 checkpoint full resume, same config (regression (c)) -----
    run_main(base + a5_resume + ["--resume", str(pr2_copy),
                                 "--max_rounds", str(total)],
             label="M6: PR-2 A5 checkpoint full resume")
    ck_m6 = torch_load_compat(synth / "pr2_resume" / "checkpoints"
                              / "task_01_end.pth")
    m6_state, a1_state = ck_m6["a5_proto_state"], ck_a1["a5_proto_state"]
    assert m6_state["version"] == a1_state["version"] == 1
    assert set(m6_state["sums"]) == set(a1_state["sums"]), \
        "M6: resumed final table keys != uninterrupted run"
    assert m6_state["counts"] == a1_state["counts"], \
        "M6: resumed final table counts != uninterrupted run"
    max_diff = max(
        float((m6_state["sums"][c] - a1_state["sums"][c]).abs().max())
        for c in a1_state["sums"])
    assert max_diff <= 1e-6, f"M6: sums max_diff {max_diff} > 1e-6"
    print(f"[PASS] M6: resumed table == uninterrupted run "
          f"({len(a1_state['sums'])} classes, counts equal, "
          f"sums max_diff={max_diff})")

    # ---- hybrid legacy: table present, hparams missing (M8) ------------
    run_main(base + a5_resume + ["--resume", str(hybrid),
                                 "--max_rounds", str(total),
                                 "--allow_proto_migration"],
             label="M8: hybrid legacy migration (table kept)")
    ck_m8 = torch_load_compat(synth / "hybrid" / "checkpoints"
                              / "task_01_end.pth")
    m8_keys = set(ck_m8["a5_proto_state"]["sums"])
    assert ck_m8["a5_proto_state"]["version"] == 1
    assert m8_keys == set(a1_state["sums"]), \
        "M8: hybrid migration must RESTORE the carried table (keys != " \
        "uninterrupted run)"
    print(f"[PASS] M8: hybrid migration restored the carried table "
          f"({len(m8_keys)} classes == uninterrupted run)")

    # ---- M9: PR-3 increment (provenance marker + poison rejection) ------
    # M9a: the flag must be a NO-OP on a full PR-2 checkpoint: lambda/tau
    # come from the checkpoint (RESUME_MUST_MATCH), the table is restored
    # and the final state is bit-identical to the uninterrupted run.
    pr2_flag = synth / "pr2_flag" / "checkpoints" / "ckpt.pth"
    pr2_flag.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(boundary, pr2_flag)
    run_main(base + a5_resume + ["--resume", str(pr2_flag),
                                 "--max_rounds", str(total),
                                 "--allow_proto_migration"],
             label="M9a: PR-2 resume WITH flag (must be a no-op)")
    ck_m9a = torch_load_compat(synth / "pr2_flag" / "checkpoints"
                               / "task_01_end.pth")
    m9a_state = ck_m9a["a5_proto_state"]
    assert float(ck_m9a["args"]["a5_proto_lambda"]) == \
        float(ck_a1["args"]["a5_proto_lambda"]), \
        "M9a: flag must not change lambda on a PR-2 checkpoint"
    assert float(ck_m9a["args"]["a5_proto_tau"]) == \
        float(ck_a1["args"]["a5_proto_tau"]), \
        "M9a: flag must not change tau on a PR-2 checkpoint"
    assert "a5_proto_migration" not in ck_m9a, \
        "M9a: a non-migrated run must NOT carry the migration marker"
    assert set(m9a_state["sums"]) == set(a1_state["sums"]) \
        and m9a_state["counts"] == a1_state["counts"]
    m9a_diff = max(
        float((m9a_state["sums"][c] - a1_state["sums"][c]).abs().max())
        for c in a1_state["sums"])
    assert m9a_diff <= 1e-6, f"M9a: sums max_diff {m9a_diff} > 1e-6"
    print(f"[PASS] M9a: PR-2 + flag == no-op (lambda/tau from checkpoint, "
          f"table bit-identical, max_diff={m9a_diff})")

    # M9e (reviewer-requested negative): the flag must NOT bypass the
    # RESUME_MUST_MATCH consistency check. validate_checkpoint_args runs
    # BEFORE the migration gate, so a PR-2 checkpoint resumed with a
    # DIFFERENT CLI lambda is rejected even under --allow_proto_migration
    # (checkpoint values are never silently overridden by CLI values).
    run_main(base + a5_resume + ["--resume", str(pr2_flag),
                                 "--max_rounds", str(total),
                                 "--a5_proto_lambda", "0.25",
                                 "--allow_proto_migration"],
             expect_fail=True, expect_msg="Checkpoint mismatch",
             label="M9e: flag cannot bypass lambda/tau mismatch rejection")

    # M9b: NaN-poisoned table is rejected fail-closed.
    import torch
    nan_ckpt = synth / "nan_table" / "checkpoints" / "ckpt.pth"
    nan_ckpt.parent.mkdir(parents=True, exist_ok=True)
    poisoned = torch_load_compat(boundary)
    first_class = sorted(poisoned["a5_proto_state"]["sums"])[0]
    poisoned["a5_proto_state"]["sums"][first_class][0] = float("nan")
    torch.save(poisoned, nan_ckpt)
    run_main(base + a5_resume + ["--resume", str(nan_ckpt),
                                 "--max_rounds", str(total)],
             expect_fail=True, expect_msg="NaN/Inf",
             label="M9b: NaN-poisoned prototype table refused")

    # M9c: migrated runs stamp EVERY checkpoint with the provenance marker.
    for name, ck in (("M5", ck_m5), ("M8", ck_m8)):
        marker = ck.get("a5_proto_migration")
        assert isinstance(marker, dict) and marker, \
            f"M9c: migrated run's checkpoint ({name}) must carry the " \
            f"a5_proto_migration provenance marker"
        assert marker["migrated_from_round"] == ge - 1 \
            and marker["migrated_at_task"] == 0, \
            f"M9c: {name} provenance fields wrong: {marker}"
    assert "a5_proto_migration" not in ck_a1, \
        "M9c: the uninterrupted run must NOT carry the marker"
    print("[PASS] M9c: migrated checkpoints carry the provenance marker "
          "(fresh-table M5 and hybrid M8); non-migrated runs do not")

    # M9d: chained resume from a MIGRATED run's checkpoint inherits the
    # marker WITHOUT the flag (provenance, not permission) and the
    # trajectory is unchanged. M5's mid-task round checkpoint (round 2 of
    # task 1, saved AFTER the migration + first task-entry sync) has the
    # full PR-2 schema, so a plain resume is legal; redoing its last
    # round must reproduce M5's final table bit-exactly.
    m5_round2 = synth / "legacy_full" / "checkpoints" / "round_0002.pth"
    assert m5_round2.is_file(), f"M9d: missing {m5_round2}"
    m5_round2_copy = synth / "m9_inherit" / "checkpoints" / "ckpt.pth"
    m5_round2_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(m5_round2, m5_round2_copy)
    run_main(base + a5_resume + ["--resume", str(m5_round2_copy),
                                 "--max_rounds", str(total),
                                 "--a5_proto_lambda", "0.25",
                                 "--a5_proto_tau", "0.11"],
             label="M9d: chained resume of migrated run (no flag)")
    ck_m9d = torch_load_compat(synth / "m9_inherit" / "checkpoints"
                               / "task_01_end.pth")
    assert isinstance(ck_m9d.get("a5_proto_migration"), dict), \
        "M9d: chained resume must INHERIT the migration marker"
    m9d_state = ck_m9d["a5_proto_state"]
    assert set(m9d_state["sums"]) == set(m5_state["sums"]) \
        and m9d_state["counts"] == m5_state["counts"], \
        "M9d: inherited-marker resume changed the table keys/counts"
    m9d_diff = max(
        float((m9d_state["sums"][c] - m5_state["sums"][c]).abs().max())
        for c in m5_state["sums"])
    assert m9d_diff == 0.0, \
        f"M9d: trajectory changed under inheritance (max_diff={m9d_diff})"
    print(f"[PASS] M9d: marker inherited without flag; trajectory "
          f"unchanged (sums max_diff={m9d_diff} vs M5)")

    print("\nRESULT: PASS (migration guard: default refusal + explicit "
          "migration + schema/version/boundary validation + FedTA/A2 and "
          "PR-2 resume regressions + provenance marker + poison rejection)")


if __name__ == "__main__":
    main()
