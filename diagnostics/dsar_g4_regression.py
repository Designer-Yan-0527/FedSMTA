"""G4 DSAR 训练级回归（RUNCOMMANDS §34.4d item 3 + §34.4 Evaluation
integrity，CIFAR B1 formal 解锁的最后前置签核）。

最短 deterministic matched OFF/ON(identity) 训练回归：

  A  g4_off_full  : DSAR OFF，连续 N 轮（官方训练路径基线）
  B  g4_on_full   : --dsar --dsar_identity（ON(identity) 测试模式：跳过
                    全部拟合、每个 broadcast 边界提交 exact-identity
                    sidecar），连续 N 轮
  C  g4_on_split  : ON(identity) 先跑 K 轮，--resume auto 续跑到 N
                    （checkpoint/resume 对应性；K 默认跨 task 边界）

G4 断言（§34.4d 冻结口径）：
  G4_TRAJECTORY   逐轮（每个 round_XXXX.pth / task_XX_end.pth /
                  latest.pth）A vs B 全字段 bitwise 相等：server model /
                  global head / server prompt / global+temp protos /
                  per-client heads / tail-anchor 全状态 / prompts /
                  protos / RNG 四流 / inner 70/30 split / outer manifest /
                  existing_class / fix_keys / completed_round / task_id。
                  （训练流 optimizer/scheduler 每轮新建、无跨轮持久
                  state——checkpoint 亦不含之；模型 + RNG + split 全等
                  即完整训练状态逐位等价。）
  G4_EVAL         accuracy_matrix.csv A vs B vs C 逐位相等（task-end
                  全任务评估经 sidecar apply(identity) 读出，含历史任务
                  路径的端到端中立性）。
  G4_RESUME       C 续跑后逐轮 checkpoint 与 A 对应轮 bitwise 相等；
                  latest.pth 全等；resume 确实发生（train.log 含
                  RESUME_CHECKPOINT）。
  G4_SEAL         封章（防假通过）：A 的 args 记录 dsar=False；B/C 记录
                  dsar=True + dsar_identity=True；A 的 client state 无
                  dsar_sidecar key（OFF checkpoint schema 逐位不变 =
                  OFF 正式训练/evaluate 未被触碰的证据）；B/C 每 client
                  携带 identity sidecar（b_active=False / l_active=False /
                  无因子 / status=IDENTITY_TEST）。

G1 写盘纪律：唯一持久写盘点 = verdict JSON（默认
<work_dir>/metrics/dsar_g4_regression.json）；三条回归 run 目录是
新建的回归草稿目录（output/regression/...），不触碰任何封板 run。

用法（GPU 机器上、repo 根目录）：
  python diagnostics/dsar_g4_regression.py \
      --data_name cifar100 --data-path ./local_datasets \
      --rounds 3 --split_at 2 --global_epoch 2 --task_num 2 \
      --local_epoch 2 --device cuda

  rounds 默认 = global_epoch + 1（跨过 task 0 末边界进入 task 1，
  使 task-end 全任务评估 + 历史任务 sidecar 读出路径都被覆盖）；
  split_at 默认 = global_epoch（resume 跨 task 边界，最难对应性场景）。
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_resume_regression import (  # noqa: E402
    REPO_ROOT as _RR, max_diff, state_dict_diff, protos_diff, deep_equal,
    hash_obj, load_matrix, matrix_diff, require_inner_split_state,
    require_outer_manifest, torch_load_compat,
)
from Models.DSAR import DSAR_VERSION  # noqa: E402  (sidecar 版本封章)

assert _RR == REPO_ROOT  # same repo, different discovery path


def parse_args():
    p = argparse.ArgumentParser("G4 DSAR training-level regression")
    p.add_argument("--data_name", default="cifar100",
                   choices=["cifar100", "ImageNet-R"])
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--rounds", default=0, type=int,
                   help="total rounds N (0 = global_epoch + 1: crosses the "
                        "task-0 end boundary into task 1)")
    p.add_argument("--split_at", default=0, type=int,
                   help="stop arm C after K rounds then resume (0 = "
                        "global_epoch: resume across the task boundary)")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=2, type=int,
                   help="keep small: this only tests DSAR neutrality")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work_dir", default="output/regression", type=Path)
    p.add_argument("--method", default="fedta")
    p.add_argument("--out_json", default=None,
                   help="default: <work_dir>/metrics/dsar_g4_regression.json")
    return p.parse_args()


def run_main(extra):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


# ---------------------------------------------------------- ckpt comparison

CLIENT_EXCLUDE_KEYS = {"dsar_sidecar"}


def client_state_diff(ca, cb):
    """Per-client full-state bitwise compare, excluding the ON-only
    dsar_sidecar key (validated separately by the seal)."""
    a_ex = {k: v for k, v in ca.items() if k not in CLIENT_EXCLUDE_KEYS}
    b_ex = {k: v for k, v in cb.items() if k not in CLIENT_EXCLUDE_KEYS}
    if deep_equal(a_ex, b_ex):
        return 0.0
    # fall back to a coarse per-key diff for diagnostics
    if a_ex.keys() != b_ex.keys():
        return float("inf")
    worst = 0.0
    for k, va in a_ex.items():
        vb = b_ex[k]
        if (isinstance(va, dict) and isinstance(vb, dict) and va
                and all(isinstance(v, torch.Tensor)
                        for v in va.values())):
            worst = max(worst, state_dict_diff(va, vb))
        elif not deep_equal(va, vb):
            return float("inf")
    return worst


def compare_ckpts(ck_a, ck_b):
    """Full checkpoint field comparison (bitwise; 0.0 / inf / bool)."""
    fields = {
        "completed_round_diff": float(abs(int(ck_a["completed_round"])
                                          - int(ck_b["completed_round"]))),
        "server_task_id_diff": float(abs(int(ck_a["server_task_id"])
                                         - int(ck_b["server_task_id"]))),
        "server_model": state_dict_diff(ck_a["server_model"],
                                        ck_b["server_model"]),
        "global_head": state_dict_diff(ck_a["global_head"],
                                       ck_b["global_head"]),
        "server_prompt": state_dict_diff(ck_a["server_prompt"],
                                          ck_b["server_prompt"]),
        "global_protos": protos_diff(ck_a["global_protos"],
                                     ck_b["global_protos"]),
        "temp_protos": protos_diff(ck_a["temp_protos"],
                                   ck_b["temp_protos"]),
        "client_state": max(client_state_diff(ca, cb)
                            for ca, cb in zip(ck_a["clients"],
                                              ck_b["clients"])),
        "rng_state": (0.0 if deep_equal(ck_a["rng_state"],
                                        ck_b["rng_state"])
                      else float("inf")),
        "data_split_state": (0.0 if deep_equal(ck_a["data_split_state"],
                                               ck_b["data_split_state"])
                             else float("inf")),
        "outer_split_manifest": (0.0 if deep_equal(
            ck_a["outer_split_manifest"], ck_b["outer_split_manifest"])
            else float("inf")),
        "existing_class": (0.0 if deep_equal(ck_a["existing_class"],
                                             ck_b["existing_class"])
                           else float("inf")),
        "fix_keys": 0.0 if (sorted(map(str, ck_a["fix_keys"]))
                            == sorted(map(str, ck_b["fix_keys"])))
        else float("inf"),
    }
    return fields


def checkpoint_inventory(run_dir):
    """Sorted list of checkpoint filenames (round_*/task_*_end/latest)."""
    return sorted(p.name for p in
                  (Path(run_dir) / "checkpoints").glob("*.pth"))


# args keys allowed to differ between the OFF and ON arms: the treatment
# flags themselves and the run identity (same class as the
# bsab_offon_regression ARG_WHITELIST precedent, §34.2 0b)
ARGS_WHITELIST = {"dsar", "dsar_identity", "run_name", "resume"}
# metrics files that must be byte-identical between arms (presence compared
# pairwise; round_metrics.jsonl is compared field-wise with the wall-clock
# "timestamp" key dropped)
METRICS_FILES = (
    "metrics/accuracy_matrix.csv",
    "metrics/train_log.csv",
    "metrics/margins.csv",
    "metrics/round_metrics.jsonl",
    "metrics/summary_metrics.csv",
)


def compare_args_whitelist(ck_a, ck_b, expect_flip):
    """checkpoint args must differ ONLY on the whitelisted keys, with the
    treatment flags flipped exactly as expected (False -> True)."""
    a, b = dict(ck_a.get("args", {})), dict(ck_b.get("args", {}))
    issues = []
    for k in set(a) | set(b):
        if a.get(k) != b.get(k) and k not in ARGS_WHITELIST:
            issues.append(f"args.{k}: {a.get(k)!r} vs {b.get(k)!r}")
    for k in expect_flip:
        if not (a.get(k) is False and b.get(k) is True):
            issues.append(f"args.{k} not flipped False->True "
                          f"({a.get(k)!r} -> {b.get(k)!r})")
    return issues


def compare_command_txt(run_a, run_b, extra_whitelist=()):
    """command.txt (append-mode) last-line flags must differ only by the
    whitelisted flags; the treatment flags must be present on arm B."""
    def last_cmd(run_dir):
        text = (Path(run_dir) / "command.txt").read_text(
            encoding="utf-8", errors="replace").strip()
        return text.splitlines()[-1] if text else ""

    def flags(cmd):
        toks = cmd.split()
        return {toks[i]: (toks[i + 1] if i + 1 < len(toks)
                          and not toks[i + 1].startswith("--") else True)
                for i, t in enumerate(toks) if t.startswith("--")}

    f_a, f_b = flags(last_cmd(run_a)), flags(last_cmd(run_b))
    whitelist = {"--dsar", "--dsar_identity", "--run_name"} | set(
        extra_whitelist)
    issues = []
    for k in set(f_a) | set(f_b):
        if f_a.get(k) != f_b.get(k) and k not in whitelist:
            issues.append(f"command.txt {k}: {f_a.get(k)!r} vs {f_b.get(k)!r}")
    for k in ("--dsar", "--dsar_identity"):
        if k not in f_b:
            issues.append(f"command.txt ON arm missing {k}")
    return issues


def compare_metrics(run_a, run_b):
    """metrics artifacts byte-identity (round_metrics.jsonl field-wise
    minus the wall-clock timestamp). Mirrors bsab_offon_regression."""
    issues = []
    for rel in METRICS_FILES:
        p_a, p_b = Path(run_a) / rel, Path(run_b) / rel
        if p_a.exists() != p_b.exists():
            issues.append(f"{rel} presence mismatch "
                          f"({p_a.exists()} vs {p_b.exists()})")
        elif not p_a.exists():
            continue
        elif rel.endswith("round_metrics.jsonl"):
            la = [json.loads(x) for x in p_a.read_text().splitlines()
                  if x.strip()]
            lb = [json.loads(x) for x in p_b.read_text().splitlines()
                  if x.strip()]
            if len(la) != len(lb):
                issues.append(f"{rel} line count {len(la)} vs {len(lb)}")
            for ra, rb in zip(la, lb):
                ra = {k: v for k, v in ra.items() if k != "timestamp"}
                rb = {k: v for k, v in rb.items() if k != "timestamp"}
                if ra != rb:
                    issues.append(f"{rel} row divergence")
                    break
        elif p_a.read_bytes() != p_b.read_bytes():
            issues.append(f"{rel} NOT byte-identical")
    return issues


def load_all_ckpts(run_dir):
    return {name: torch_load_compat(Path(run_dir) / "checkpoints" / name)
            for name in checkpoint_inventory(run_dir)}


# ------------------------------------------------------------------ seals

def check_seal(ckpts, run_dir, expect_dsar, expect_identity):
    """False-pass guards: the runs must really have run the intended arms,
    and the dsar_sidecar schema must match the expected deployment shape."""
    issues = []
    for name, ck in sorted(ckpts.items()):
        args = ck.get("args", {})
        if bool(args.get("dsar", False)) is not expect_dsar:
            issues.append(f"{name}: args.dsar={args.get('dsar')} "
                          f"(expected {expect_dsar})")
        if expect_dsar and (bool(args.get("dsar_identity", False))
                            is not expect_identity):
            issues.append(f"{name}: args.dsar_identity="
                          f"{args.get('dsar_identity')}")
        for c in ck["clients"]:
            has_key = "dsar_sidecar" in c
            if expect_dsar and not has_key:
                issues.append(f"{name}: client {c['id']} missing "
                               "dsar_sidecar key")
            if (not expect_dsar) and has_key:
                issues.append(f"{name}: client {c['id']} carries "
                               "dsar_sidecar (OFF schema must be "
                               "unchanged)")
            if expect_dsar and has_key and c["dsar_sidecar"] is not None:
                sc = c["dsar_sidecar"]
                if (sc.get("b_active") or sc.get("l_active")
                        or sc.get("b") is not None
                        or sc.get("l") is not None):
                    issues.append(f"{name}: client {c['id']} sidecar is "
                                  "not exact-identity")
                if sc.get("version") != DSAR_VERSION:
                    issues.append(f"{name}: client {c['id']} sidecar "
                                  f"version {sc.get('version')} != "
                                  f"{DSAR_VERSION}")
                if (sc.get("meta", {}).get("status")
                        not in ("IDENTITY_TEST", "NO_BIRTH_STATE")):
                    issues.append(f"{name}: client {c['id']} sidecar "
                                  f"status={sc.get('meta', {}).get('status')}")
    # false-pass guard: at the final checkpoint the ON arm must carry a
    # committed (non-None) sidecar for every client -- the first broadcast
    # of round 0 commits before the first save
    if expect_dsar and "latest.pth" in ckpts:
        for c in ckpts["latest.pth"]["clients"]:
            if c.get("dsar_sidecar") is None:
                issues.append(f"latest.pth: client {c['id']} sidecar is "
                              "None (never committed)")
    return issues


def main():
    args = parse_args()
    rounds = (args.rounds if args.rounds > 0
              else args.global_epoch + 1)
    split_at = (args.split_at if args.split_at > 0
                else args.global_epoch)
    if not (0 < split_at < rounds):
        raise SystemExit(f"--split_at must be in (0, {rounds})")
    if rounds > args.task_num * args.global_epoch:
        raise SystemExit(
            f"--rounds {rounds} exceeds task_num*global_epoch "
            f"({args.task_num * args.global_epoch})")

    common = ["--data_name", args.data_name, "--data-path", args.data_path,
              "--seed", str(args.seed), "--device", args.device,
              "--global_epoch", str(args.global_epoch),
              "--task_num", str(args.task_num),
              "--local_epoch", str(args.local_epoch),
              "--method", args.method, "--deterministic",
              "--save_every", "1", "--output_dir", str(args.work_dir)]

    runs = {}
    for arm in ("g4_off_full", "g4_on_full", "g4_on_split"):
        run_dir = (args.work_dir / args.data_name / args.method / arm
                   / f"seed_{args.seed}")
        runs[arm] = run_dir
        if run_dir.exists():
            print(f"[cleanup] removing previous test run dir {run_dir}")
            shutil.rmtree(run_dir)

    dsar_on = ["--dsar", "--dsar_identity"]

    # A: OFF continuous (official path baseline)
    run_main(common + ["--run_name", "g4_off_full",
                       "--max_rounds", str(rounds)])
    # B: ON(identity) continuous
    run_main(common + ["--run_name", "g4_on_full",
                       "--max_rounds", str(rounds)] + dsar_on)
    # C: ON(identity) split + resume (resume across the task boundary)
    run_main(common + ["--run_name", "g4_on_split",
                       "--max_rounds", str(split_at)] + dsar_on)
    run_main(common + ["--run_name", "g4_on_split",
                       "--max_rounds", str(rounds),
                       "--resume", "auto"] + dsar_on)

    ckpts = {arm: load_all_ckpts(runs[arm]) for arm in runs}

    report = []          # (name, ok, detail)
    all_ok = True

    def add(name, ok, detail=""):
        nonlocal all_ok
        all_ok &= bool(ok)
        report.append((name, bool(ok), detail))

    # ---- G4_SEAL ------------------------------------------------------
    seal_issues = (check_seal(ckpts["g4_off_full"], runs["g4_off_full"],
                              expect_dsar=False, expect_identity=False)
                   + check_seal(ckpts["g4_on_full"], runs["g4_on_full"],
                                expect_dsar=True, expect_identity=True)
                   + check_seal(ckpts["g4_on_split"], runs["g4_on_split"],
                                expect_dsar=True, expect_identity=True))
    add("G4_SEAL_args_and_schema", not seal_issues,
        "; ".join(seal_issues[:5]))

    # resume really happened
    train_log = runs["g4_on_split"] / "logs" / "train.log"
    resumed = (train_log.is_file()
               and "RESUME_CHECKPOINT" in train_log.read_text(
                   encoding="utf-8", errors="replace"))
    add("G4_SEAL_resume_happened", resumed,
        "train.log lacks RESUME_CHECKPOINT" if not resumed else "")

    # ---- inventory + per-round trajectory (A vs B, A vs C) ------------
    inv_a = checkpoint_inventory(runs["g4_off_full"])
    inv_b = checkpoint_inventory(runs["g4_on_full"])
    inv_c = checkpoint_inventory(runs["g4_on_split"])
    add("G4_TRAJECTORY_inventory_off_vs_on",
        inv_a == inv_b, f"{len(inv_a)} vs {len(inv_b)} files")
    add("G4_RESUME_inventory_off_vs_split",
        inv_a == inv_c, f"{len(inv_a)} vs {len(inv_c)} files")

    per_round_ab, per_round_ac = {}, {}
    for name in inv_a:
        f_ab = compare_ckpts(ckpts["g4_off_full"][name],
                             ckpts["g4_on_full"][name])
        per_round_ab[name] = {k: (None if v == float("inf") else v)
                               for k, v in f_ab.items()}
        add(f"G4_TRAJECTORY[{name}]", all(v == 0.0 for v in f_ab.values()),
            "; ".join(k for k, v in f_ab.items() if v != 0.0))
        if name in ckpts["g4_on_split"]:
            f_ac = compare_ckpts(ckpts["g4_off_full"][name],
                                 ckpts["g4_on_split"][name])
            per_round_ac[name] = {
                k: (None if v == float("inf") else v)
                for k, v in f_ac.items()}
            add(f"G4_RESUME[{name}]",
                all(v == 0.0 for v in f_ac.values()),
                "; ".join(k for k, v in f_ac.items() if v != 0.0))

    # split-state / manifest false-pass guards on the OFF baseline
    expected_tasks = list(range((rounds - 1) // args.global_epoch + 1))
    try:
        require_inner_split_state(ckpts["g4_off_full"]["latest.pth"],
                                  str(runs["g4_off_full"]), expected_tasks)
        require_outer_manifest(ckpts["g4_off_full"]["latest.pth"],
                               str(runs["g4_off_full"]))
        add("G4_SEAL_split_state_present", True)
    except AssertionError as e:
        add("G4_SEAL_split_state_present", False, str(e))

    # ---- G4_EVAL: accuracy matrices + metrics artifacts ----------------
    reached_task_end = rounds >= args.global_epoch
    if reached_task_end:
        mats = {arm: load_matrix(runs[arm]) for arm in runs}
        missing = [a for a, m in mats.items() if m is None]
        if missing:
            add("G4_EVAL_accuracy_matrix", False,
                f"missing accuracy_matrix.csv: {missing}")
        else:
            d_ab = matrix_diff(mats["g4_off_full"], mats["g4_on_full"])
            d_ac = matrix_diff(mats["g4_off_full"], mats["g4_on_split"])
            add("G4_EVAL_accuracy_matrix",
                d_ab == 0.0 and d_ac == 0.0,
                f"off_vs_on={d_ab}, off_vs_split={d_ac}")
        # metrics artifacts byte-identity (bsab_offon_regression precedent)
        m_ab = compare_metrics(runs["g4_off_full"], runs["g4_on_full"])
        m_ac = compare_metrics(runs["g4_off_full"], runs["g4_on_split"])
        add("G4_EVAL_metrics_artifacts", not (m_ab or m_ac),
            "; ".join((m_ab + m_ac)[:5]))
    else:
        add("G4_EVAL_accuracy_matrix", True,
            "skipped: run did not reach a task end")
        add("G4_EVAL_metrics_artifacts", True,
            "skipped: run did not reach a task end")

    # ---- args / command.txt whitelists (treatment-flip seal) -----------
    flip = ["dsar", "dsar_identity"]
    a_args = (compare_args_whitelist(
        ckpts["g4_off_full"]["latest.pth"], ckpts["g4_on_full"]["latest.pth"],
        flip)
        + compare_args_whitelist(
            ckpts["g4_off_full"]["latest.pth"],
            ckpts["g4_on_split"]["latest.pth"], flip))
    add("G4_SEAL_args_whitelist", not a_args, "; ".join(a_args[:5]))
    c_cmd = (compare_command_txt(runs["g4_off_full"], runs["g4_on_full"])
             + compare_command_txt(runs["g4_off_full"], runs["g4_on_split"],
                                   extra_whitelist=("--resume",)))
    add("G4_SEAL_command_txt", not c_cmd, "; ".join(c_cmd[:5]))

    # ---- verdict -------------------------------------------------------
    result = "PASS" if all_ok else "FAIL"
    print("\n===== G4 DSAR training-level regression report =====")
    for name, ok, detail in report:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:44s} {detail}")

    out_json = (Path(args.out_json) if args.out_json else
                Path(args.work_dir) / "metrics" / "dsar_g4_regression.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    verification = {
        "test": "dsar_g4_off_vs_on_identity_training_regression",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meaning": "DSAR ON(identity) must be bit-identical to OFF across "
                   "the whole training trajectory (per-round model/RNG/"
                   "split states), the official evaluation, and "
                   "checkpoint/resume correspondence",
        "result": result,
        "config": {
            "data_name": args.data_name, "method": args.method,
            "seed": args.seed, "rounds": rounds, "split_at": split_at,
            "global_epoch": args.global_epoch, "task_num": args.task_num,
            "local_epoch": args.local_epoch, "deterministic": True,
        },
        "runs": {arm: str(runs[arm]) for arm in runs},
        "optimizer_scheduler_note": (
            "training optimizers are re-created per round/phase inside "
            "Client_DF (torch.optim.Adam constructed in the train/"
            "phase functions; no cross-round optimizer or scheduler "
            "state exists and checkpoints carry none) -- bitwise "
            "model+RNG+split equality per round is the complete "
            "training-state equivalence"),
        "g4_trajectory_off_vs_on": per_round_ab,
        "g4_resume_off_vs_split": per_round_ac,
        "reports": [{"name": n, "ok": ok, "detail": d}
                    for n, ok, d in report],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, default=str)
    print(f"\nsaved {out_json}")
    print("\nRESULT:", result)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
