"""G4 DSAR 训练级回归（RUNCOMMANDS §34.4d item 3 + §34.4 Evaluation
integrity；2026-11-08 GPT §34.4d G4 执行审核裁定分级验收）。

matched deterministic 训练回归，五个臂：

  A  g4_off_full     : DSAR OFF，连续 N 轮（官方训练路径基线）
  B  g4_on_full      : --dsar --dsar_identity（ON(identity)：跳过全部
                       拟合、每 broadcast 边界提交 exact-identity
                       sidecar），连续 N 轮
  C  g4_on_split     : ON(identity) 先跑 K 轮，--resume auto 续跑到 N
  D  g4_on_real_full : --dsar（ON(real-fit)：真实 L 训练 + pre/post
                       z±采集 + ridge/SVD r8 + identity guard + 非恒等
                       sidecar 提交），连续 N 轮
  E  g4_on_real_split : ON(real-fit) 先跑 K 轮，--resume auto 续跑到 N

G4-A 断言（identity 路径，§34.4d 冻结口径）：
  G4_TRAJECTORY   逐轮（每个 round_XXXX.pth / task_XX_end.pth /
                  latest.pth）A vs B 全字段 bitwise 相等：server model /
                  global head / server prompt / global+temp protos /
                  per-client heads / tail-anchor 全状态 / prompts /
                  protos / RNG 四流 / inner 70/30 split / outer manifest /
                  existing_class / fix_keys / completed_round / task_id。
                  （训练流 optimizer/scheduler 每轮新建、无跨轮持久
                  state——checkpoint 亦不含之；模型 + RNG + split 全等
                  即完整训练状态逐位等价。）
  G4_EVAL         accuracy_matrix.csv A vs B vs C 逐位相等 + metrics
                  文件逐字节一致（task-end 全任务评估经 sidecar
                  apply(identity) 读出的端到端中立性）。
  G4_RESUME       C 续跑后逐轮 checkpoint 与 A 对应轮 bitwise 相等；
                  latest.pth 全等；resume 确实发生。
  G4_SEAL         封章：A 的 args 记录 dsar=False；B/C 记录 dsar=True +
                  dsar_identity=True；A 的 client state 无 dsar_sidecar
                  key（OFF checkpoint schema 逐位不变）；B/C 每 client
                  携带 identity sidecar；args/command.txt 白名单
                  （treatment-flip）。

G4-B 断言（real-fit 路径，2026-11-08 审核补齐项）：
  G4B_TRAJECTORY  ON(real-fit) 逐轮 checkpoint 与 A bitwise 相等——真实
                  拟合（L 100-epoch 训练、pre/post 前向、ridge、SVD、
                  identity guard、非恒等 sidecar 提交）不改 FedTA 训练
                  轨迹（四流 RNG snapshot/restore + head/prompt/mode
                  状态恢复的跨轮完整验证；DSAR logits / accuracy 允许
                  不同，不作为 gate）。
  G4B_RESUME      E 续跑后逐轮与 A bitwise 相等；E 与 D 的
                  accuracy_matrix 相等（同臂 resume 对账）；resume 确实
                  发生。
  G4B_SEAL        args 记录 dsar=True / dsar_identity=False；每 client
                  sidecar 版本封章 / r8 因子形状与有限性 / active 旗标
                  一致性；sidecar 提交时点正确（meta.round_id ==
                  completed_round - 1，即轮末广播提交）；全程至少一个
                  FITTED 在线状态（B=FITTED 或 L=FITTED；审核要求）。

裁决分级（§34.4d 执行审核）：
  G4_IDENTITY_PASS = A/B/C 三臂全绿（不得扩展解释为 FULL）
  G4_FULL_PASS     = G4_IDENTITY_PASS + G4-B 全绿 + ≥1 FITTED
  仅 G4_FULL_PASS 解锁 CIFAR B1 formal。

G1 写盘纪律：唯一持久写盘点 = verdict JSON（默认
<work_dir>/metrics/dsar_g4_regression.json）；五条回归 run 目录是新建
的回归草稿目录（output/regression/...），不触碰任何封板 run。identity
段结果先落盘，随后才跑 real 臂（real 臂若 fail-closed 崩溃不丢 identity
证据）。

用法（GPU 机器上、repo 根目录）：
  python diagnostics/dsar_g4_regression.py \
      --data_name cifar100 --data-path ./local_datasets \
      --rounds 3 --split_at 2 --global_epoch 2 --task_num 2 \
      --local_epoch 2 --device cuda

  rounds 默认 = global_epoch + 1（跨过 task 0 末边界进入 task 1，
  使 task-end 全任务评估 + 历史任务 sidecar 读出路径都被覆盖）；
  split_at 默认 = global_epoch（resume 跨 task 边界，最难对应性场景）。
  --skip_real 只跑 G4-A 三臂（verdict 分级 G4_IDENTITY）。
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
from Models.DSAR import DSAR_VERSION, DSAR_RANK, FEAT_DIM  # noqa: E402

assert _RR == REPO_ROOT  # same repo, different discovery path

IDENTITY_ARMS = ("g4_off_full", "g4_on_full", "g4_on_split")
REAL_ARMS = ("g4_on_real_full", "g4_on_real_split")


def parse_args():
    p = argparse.ArgumentParser("G4 DSAR training-level regression")
    p.add_argument("--data_name", default="cifar100",
                   choices=["cifar100", "ImageNet-R"])
    p.add_argument("--data-path", "--data_path", dest="data_path",
                   default="./local_datasets")
    p.add_argument("--rounds", default=0, type=int,
                   help="total rounds N (0 = global_epoch + 1: crosses the "
                        "task-0 end boundary into task 1)")
    p.add_argument("--split_at", default=0, type=int,
                   help="stop the split arms after K rounds then resume "
                        "(0 = global_epoch: resume across the task boundary)")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=2, type=int,
                   help="keep small: this only tests DSAR neutrality")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work_dir", default="output/regression", type=Path)
    p.add_argument("--method", default="fedta")
    p.add_argument("--skip_real", action="store_true",
                   help="run only the G4-A identity arms (verdict scope "
                        "G4_IDENTITY, not sufficient for B1 unlock)")
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
    dsar_sidecar key (validated separately by the seals)."""
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


def load_all_ckpts(run_dir):
    return {name: torch_load_compat(Path(run_dir) / "checkpoints" / name)
            for name in checkpoint_inventory(run_dir)}


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
    named treatment flags flipped exactly as expected."""
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


def compare_command_txt(run_a, run_b, required=("--dsar", "--dsar_identity"),
                         extra_whitelist=()):
    """command.txt (append-mode) last-line flags must differ only by the
    whitelisted flags; the given treatment flags must be present on arm B."""
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
    whitelist = ({"--dsar", "--dsar_identity", "--run_name"}
                 | set(extra_whitelist))
    issues = []
    for k in set(f_a) | set(f_b):
        if f_a.get(k) != f_b.get(k) and k not in whitelist:
            issues.append(f"command.txt {k}: {f_a.get(k)!r} vs {f_b.get(k)!r}")
    for k in required:
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


# ------------------------------------------------------------------ seals

def check_seal(ckpts, expect_dsar, expect_identity):
    """G4-A false-pass guards for the identity arms: the runs must really
    have run the intended arms, and the dsar_sidecar schema must match the
    expected deployment shape."""
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


def sidecar_factor_issues(sc):
    """Structural validation of a committed (possibly non-identity)
    sidecar state dict: version seal, r8 factor shapes, finiteness and
    active-flag consistency."""
    issues = []
    if sc.get("version") != DSAR_VERSION:
        issues.append(f"version {sc.get('version')} != {DSAR_VERSION}")
    for key in ("b", "l"):
        d = sc.get(key)
        active = bool(sc.get(key + "_active"))
        if d is None:
            if active:
                issues.append(f"{key}_active=True but no {key} factors")
            continue
        if not active:
            issues.append(f"{key} factors present but {key}_active=False")
        for name, shape in (("M1", (FEAT_DIM, DSAR_RANK)),
                            ("M2", (FEAT_DIM, DSAR_RANK)),
                            ("bias", (FEAT_DIM,))):
            t = d.get(name)
            if t is None or tuple(t.shape) != shape:
                issues.append(f"{key}.{name} bad shape "
                              f"{None if t is None else tuple(t.shape)}")
            elif not bool(torch.isfinite(t).all()):
                issues.append(f"{key}.{name} non-finite")
    return issues


def check_seal_real(ckpts):
    """G4-B seal for the ON(real-fit) arms. Returns (issues, status_counts,
    n_fitted).

    Verified per checkpoint: args record dsar=True / dsar_identity=False;
    every client carries a structurally valid sidecar; the commit timing
    is right (sidecar is committed at the round-end broadcast, so
    meta.round_id == completed_round - 1); entry is a broadcast path."""
    issues, status_counts, n_fitted = [], {}, 0
    for name, ck in sorted(ckpts.items()):
        args = ck.get("args", {})
        if args.get("dsar") is not True:
            issues.append(f"{name}: args.dsar={args.get('dsar')}")
        if args.get("dsar_identity") is not False:
            issues.append(f"{name}: args.dsar_identity="
                          f"{args.get('dsar_identity')}")
        cr = int(ck["completed_round"])
        for c in ck["clients"]:
            sc = c.get("dsar_sidecar")
            if sc is None:
                issues.append(f"{name}: client {c['id']} sidecar is None")
                continue
            issues += [f"{name}:c{c['id']}: {i}"
                       for i in sidecar_factor_issues(sc)]
            meta = sc.get("meta", {})
            if int(meta.get("round_id", -1)) != cr - 1:
                issues.append(f"{name}: client {c['id']} sidecar "
                              f"round_id={meta.get('round_id')} != "
                              f"completed_round-1 ({cr - 1})")
            if meta.get("entry") not in ("broadcast", "broadcast_no_test"):
                issues.append(f"{name}: client {c['id']} entry="
                              f"{meta.get('entry')}")
            key = "B=%s/L=%s" % (meta.get("b_status"), meta.get("l_status"))
            status_counts[key] = status_counts.get(key, 0) + 1
            if (meta.get("b_status") == "FITTED"
                    or meta.get("l_status") == "FITTED"):
                n_fitted += 1
    return issues, status_counts, n_fitted


def resume_happened(run_dir):
    train_log = Path(run_dir) / "logs" / "train.log"
    return (train_log.is_file()
            and "RESUME_CHECKPOINT" in train_log.read_text(
                encoding="utf-8", errors="replace"))


def per_round_compare(ck_a, ckpts_b, inv_a, prefix, add):
    """Per-round bitwise comparison of every checkpoint of run B against
    the matching OFF checkpoint; also reports the field-level diffs."""
    per_round = {}
    for name in inv_a:
        if name not in ckpts_b:
            continue
        f = compare_ckpts(ck_a[name], ckpts_b[name])
        per_round[name] = {k: (None if v == float("inf") else v)
                           for k, v in f.items()}
        add(f"{prefix}[{name}]", all(v == 0.0 for v in f.values()),
            "; ".join(k for k, v in f.items() if v != 0.0))
    return per_round


# ------------------------------------------------------------------ driver

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

    arms = list(IDENTITY_ARMS) + ([] if args.skip_real else list(REAL_ARMS))
    runs = {}
    for arm in arms:
        run_dir = (args.work_dir / args.data_name / args.method / arm
                   / f"seed_{args.seed}")
        runs[arm] = run_dir
        if run_dir.exists():
            print(f"[cleanup] removing previous test run dir {run_dir}")
            shutil.rmtree(run_dir)

    dsar_identity_on = ["--dsar", "--dsar_identity"]
    dsar_real_on = ["--dsar"]

    # ================= G4-A: identity arms =================
    # A: OFF continuous (official path baseline)
    run_main(common + ["--run_name", "g4_off_full",
                       "--max_rounds", str(rounds)])
    # B: ON(identity) continuous
    run_main(common + ["--run_name", "g4_on_full",
                       "--max_rounds", str(rounds)] + dsar_identity_on)
    # C: ON(identity) split + resume (resume across the task boundary)
    run_main(common + ["--run_name", "g4_on_split",
                       "--max_rounds", str(split_at)] + dsar_identity_on)
    run_main(common + ["--run_name", "g4_on_split",
                       "--max_rounds", str(rounds),
                       "--resume", "auto"] + dsar_identity_on)

    ckpts = {arm: load_all_ckpts(runs[arm]) for arm in IDENTITY_ARMS}

    report = []          # (name, ok, detail)

    def add(name, ok, detail=""):
        report.append((name, bool(ok), detail))

    # ---- G4_SEAL ------------------------------------------------------
    seal_issues = (check_seal(ckpts["g4_off_full"],
                              expect_dsar=False, expect_identity=False)
                   + check_seal(ckpts["g4_on_full"],
                                expect_dsar=True, expect_identity=True)
                   + check_seal(ckpts["g4_on_split"],
                                expect_dsar=True, expect_identity=True))
    add("G4_SEAL_args_and_schema", not seal_issues,
        "; ".join(seal_issues[:5]))

    add("G4_SEAL_resume_happened",
        resume_happened(runs["g4_on_split"]),
        "train.log lacks RESUME_CHECKPOINT"
        if not resume_happened(runs["g4_on_split"]) else "")

    # ---- inventory + per-round trajectory (A vs B, A vs C) ------------
    inv_a = checkpoint_inventory(runs["g4_off_full"])
    inv_b = checkpoint_inventory(runs["g4_on_full"])
    inv_c = checkpoint_inventory(runs["g4_on_split"])
    add("G4_TRAJECTORY_inventory_off_vs_on",
        inv_a == inv_b, f"{len(inv_a)} vs {len(inv_b)} files")
    add("G4_RESUME_inventory_off_vs_split",
        inv_a == inv_c, f"{len(inv_a)} vs {len(inv_c)} files")

    per_round_ab = per_round_compare(
        ckpts["g4_off_full"], ckpts["g4_on_full"], inv_a,
        "G4_TRAJECTORY", add)
    per_round_ac = per_round_compare(
        ckpts["g4_off_full"], ckpts["g4_on_split"], inv_a,
        "G4_RESUME", add)

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
        mats = {arm: load_matrix(runs[arm]) for arm in IDENTITY_ARMS}
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
        ckpts["g4_off_full"]["latest.pth"],
        ckpts["g4_on_full"]["latest.pth"], flip)
        + compare_args_whitelist(
            ckpts["g4_off_full"]["latest.pth"],
            ckpts["g4_on_split"]["latest.pth"], flip))
    add("G4_SEAL_args_whitelist", not a_args, "; ".join(a_args[:5]))
    c_cmd = (compare_command_txt(runs["g4_off_full"], runs["g4_on_full"])
             + compare_command_txt(runs["g4_off_full"], runs["g4_on_split"],
                                   extra_whitelist=("--resume",)))
    add("G4_SEAL_command_txt", not c_cmd, "; ".join(c_cmd[:5]))

    identity_ok = all(ok for _, ok, _ in report)

    out_json = (Path(args.out_json) if args.out_json else
                Path(args.work_dir) / "metrics" / "dsar_g4_regression.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def write_verdict(scope, identity_ok_, full_ok):
        result = identity_ok_ and (full_ok is None or full_ok)
        verification = {
            "test": "dsar_g4_off_vs_on_training_regression",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meaning": "DSAR ON must be bit-identical to OFF across the "
                       "whole training trajectory (per-round model/RNG/"
                       "split states); identity arms prove sidecar "
                       "no-load neutrality, real-fit arms prove that "
                       "actual L/B fitting leaves the FedTA trajectory "
                       "untouched (logits/accuracy may differ)",
            "scope": scope,
            "identity_result": ("PASS" if identity_ok_ else "FAIL"),
            "full_result": (None if full_ok is None
                            else ("PASS" if full_ok else "FAIL")),
            "result": ("PASS" if result else "FAIL"),
            "config": {
                "data_name": args.data_name, "method": args.method,
                "seed": args.seed, "rounds": rounds, "split_at": split_at,
                "global_epoch": args.global_epoch, "task_num": args.task_num,
                "local_epoch": args.local_epoch, "deterministic": True,
                "skip_real": args.skip_real,
            },
            "runs": {arm: str(runs[arm]) for arm in arms},
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
        if scope == "G4_FULL":
            verification["g4b_trajectory_off_vs_real"] = per_round_da
            verification["g4b_resume_off_vs_real_split"] = per_round_ea
            verification["g4b_sidecar_status_counts"] = real_status_counts
            verification["g4b_real_vs_off_accuracy_diff"] = real_matrix_diff
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(verification, f, indent=2, default=str)
        return result

    # identity verdict lands on disk BEFORE the real arms run (a real-arm
    # fail-closed crash must not lose the identity evidence)
    write_verdict("G4_IDENTITY", identity_ok, None)

    # ================= G4-B: real-fit arms =================
    if args.skip_real:
        print("\n===== G4 DSAR training-level regression report "
              "(identity arms only) =====")
        for name, ok, detail in report:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
        print(f"\nsaved {out_json}")
        print("\nG4_IDENTITY_RESULT:", "PASS" if identity_ok else "FAIL")
        sys.exit(0 if identity_ok else 1)

    # D: ON(real-fit) continuous
    run_main(common + ["--run_name", "g4_on_real_full",
                       "--max_rounds", str(rounds)] + dsar_real_on)
    # E: ON(real-fit) split + resume (non-identity sidecar resume check)
    run_main(common + ["--run_name", "g4_on_real_split",
                       "--max_rounds", str(split_at)] + dsar_real_on)
    run_main(common + ["--run_name", "g4_on_real_split",
                       "--max_rounds", str(rounds),
                       "--resume", "auto"] + dsar_real_on)

    ckpts_real = {arm: load_all_ckpts(runs[arm]) for arm in REAL_ARMS}
    real_status_counts = {}
    per_round_da, per_round_ea = {}, {}
    real_matrix_diff = None

    # ---- G4B seals -----------------------------------------------------
    d_issues, d_counts, d_fitted = check_seal_real(
        ckpts_real["g4_on_real_full"])
    e_issues, e_counts, e_fitted = check_seal_real(
        ckpts_real["g4_on_real_split"])
    real_status_counts = {"g4_on_real_full": d_counts,
                          "g4_on_real_split": e_counts}
    add("G4B_SEAL_args_schema_timing",
        not (d_issues or e_issues),
        "; ".join((d_issues + e_issues)[:5]))
    add("G4B_SEAL_fitted_online", (d_fitted + e_fitted) > 0,
        f"fitted sidecar commits: real={d_fitted}, real_split={e_fitted}; "
        f"statuses real={d_counts}")
    add("G4B_SEAL_resume_happened",
        resume_happened(runs["g4_on_real_split"]),
        "train.log lacks RESUME_CHECKPOINT"
        if not resume_happened(runs["g4_on_real_split"]) else "")

    # ---- G4B trajectory / resume --------------------------------------
    inv_d = checkpoint_inventory(runs["g4_on_real_full"])
    inv_e = checkpoint_inventory(runs["g4_on_real_split"])
    add("G4B_TRAJECTORY_inventory_off_vs_real",
        inv_a == inv_d, f"{len(inv_a)} vs {len(inv_d)} files")
    add("G4B_RESUME_inventory_off_vs_real_split",
        inv_a == inv_e, f"{len(inv_a)} vs {len(inv_e)} files")

    per_round_da = per_round_compare(
        ckpts["g4_off_full"], ckpts_real["g4_on_real_full"], inv_a,
        "G4B_TRAJECTORY", add)
    per_round_ea = per_round_compare(
        ckpts["g4_off_full"], ckpts_real["g4_on_real_split"], inv_a,
        "G4B_RESUME", add)

    # ---- G4B eval: real vs OFF diff is diagnostic-only (allowed);
    #      real split vs real continuous must be identical ----------------
    if reached_task_end:
        mat_d = load_matrix(runs["g4_on_real_full"])
        mat_e = load_matrix(runs["g4_on_real_split"])
        mat_a = load_matrix(runs["g4_off_full"])
        if mat_d is None or mat_e is None:
            add("G4B_EVAL_real_resume_matrix", False,
                "missing accuracy_matrix.csv on real arms")
        else:
            real_matrix_diff = (None if mat_a is None
                                else matrix_diff(mat_a, mat_d))
            d_de = matrix_diff(mat_d, mat_e)
            add("G4B_EVAL_real_resume_matrix", d_de == 0.0,
                f"real_vs_real_split={d_de} (must be 0); "
                f"diagnostic real_vs_off={real_matrix_diff} (allowed "
                "to differ: sidecar readout correction is live)")
    else:
        add("G4B_EVAL_real_resume_matrix", True,
            "skipped: run did not reach a task end")

    # ---- G4B args / command.txt whitelists ------------------------------
    d_args = (compare_args_whitelist(
        ckpts["g4_off_full"]["latest.pth"],
        ckpts_real["g4_on_real_full"]["latest.pth"], ["dsar"])
        + compare_args_whitelist(
            ckpts["g4_off_full"]["latest.pth"],
            ckpts_real["g4_on_real_split"]["latest.pth"], ["dsar"]))
    add("G4B_SEAL_args_whitelist", not d_args, "; ".join(d_args[:5]))
    d_cmd = (compare_command_txt(runs["g4_off_full"],
                                 runs["g4_on_real_full"],
                                 required=("--dsar",))
             + compare_command_txt(runs["g4_off_full"],
                                   runs["g4_on_real_split"],
                                   required=("--dsar",),
                                   extra_whitelist=("--resume",)))
    add("G4B_SEAL_command_txt", not d_cmd, "; ".join(d_cmd[:5]))

    full_ok = identity_ok and all(
        ok for name, ok, _ in report if name.startswith("G4B"))

    print("\n===== G4 DSAR training-level regression report =====")
    for name, ok, detail in report:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")

    result = write_verdict("G4_FULL", identity_ok, full_ok)
    print(f"\nsaved {out_json}")
    print("\nG4_IDENTITY_RESULT:", "PASS" if identity_ok else "FAIL")
    print("G4_FULL_RESULT:", "PASS" if full_ok else "FAIL")
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
