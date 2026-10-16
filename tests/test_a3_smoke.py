"""A3-KL v1 smoke test (GPU, short runs).

Verifies the eight properties required before any long A3 run:

  S1. A3 OFF schema invariance: a plain fedta run still succeeds and its
      checkpoints carry NO a3_state key (every A3 code path is gated on
      method == 'fedta_a3'). NOT a trajectory-equivalence test — that is
      the T0/Gate-0 regression suite's job.
  S2. task 0: A3 inactive — task_00_end.pth a3_state has an empty bank and
      the log prints the task-0 inactive line (compat loss never added).
  S3. task 1+: the reference bank exists (256 indices, teacher probs
      [256, 200]) and the per-epoch logged raw compat KL is > 0.
  S4. gradient isolation: a compat-only backward gives
      key.grad != 0 while anchor / head / vit prompt / stored prompt
      parameters all keep grad == None (compatibility gradient reaches ONLY
      the key).
  S5. sampling convention: the in-process bank indices equal the
      a3_offline_calibration convention (canonical class-sorted pool,
      torch.Generator seeded 9700 + 100*client_id + task_id).
  S6. resume: resuming mid-task from latest.pth restores the SAME reference
      bank (indices equal, teacher probs bit-identical); no re-sampling.
  S7. KL direction: with the key perturbed off the teacher fixed point, the
      compat loss equals the explicit forward KL sum(r_old*(log r_old -
      log r_new))/B and is NOT the reverse KL — S3's "> 0" alone cannot
      distinguish the two directions.
  S8. fetch immutability: _a3_fetch_ref_images reads surro.data[raw_idx]
      (byte-equal to a direct read) and is unchanged after getTrainData()
      re-orders the mutable TrainData buffer (as kd_fusion_prompt does
      every round).

Usage (from repo root, on the server):
  python tests/test_a3_smoke.py
  python tests/test_a3_smoke.py --data_path ./local_datasets --device cuda
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_resume_regression import torch_load_compat  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("A3-KL v1 smoke test")
    p.add_argument("--data_name", default="cifar100")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global_epoch", default=2, type=int,
                   help="rounds per task (2 => task 1 starts at round 2)")
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=1, type=int)
    p.add_argument("--private_class_num", default=0, type=int,
                   help="0 = config default (15). Smaller values shrink "
                        "the per-client data and speed the smoke run up.")
    p.add_argument("--work_dir", default="output/regression_a3", type=Path)
    return p.parse_args()


def run_main(extra, label=""):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print((proc.stdout or "")[-3000:])
        print((proc.stderr or "")[-3000:])
        raise RuntimeError(f"{label}: command failed")
    return proc


def compat_kl_values(stdout):
    """Raw compat KL values from the per-epoch [A3] log lines."""
    vals = []
    for m in re.finditer(r"\[A3\] client \d+ task (\d+) epoch \d+: "
                         r"base_loss=[-\d.]+ compat_kl=([\d.eE+-]+)",
                         stdout):
        vals.append((int(m.group(1)), float(m.group(2))))
    return vals


def main():
    args = parse_args()
    ge = args.global_epoch
    assert ge >= 2, "need a genuine mid-task cut inside task 1"

    run_a3 = args.work_dir / args.data_name / "fedta_a3" / "a3_smoke" / f"seed_{args.seed}"
    run_ft = args.work_dir / args.data_name / "fedta" / "a3_smoke_ft" / f"seed_{args.seed}"
    for run in (run_a3, run_ft):
        if run.exists():
            print(f"[cleanup] removing previous test run dir {run}")
            shutil.rmtree(run)

    base = ["--data_name", args.data_name, "--data_path", args.data_path,
            "--seed", str(args.seed), "--device", args.device,
            "--global_epoch", str(ge), "--task_num", str(args.task_num),
            "--local_epoch", str(args.local_epoch),
            "--deterministic", "--save_every", "1",
            "--output_dir", str(args.work_dir)]
    if args.private_class_num > 0:
        base += ["--private_class_num", str(args.private_class_num)]

    # ---- run A: fedta_a3 crossing into task 1 (>=1 round of task 1) ----
    # stop after the FIRST round of task 1 (round 2*ge-2): mid-task-1 cut
    n_rounds_a = 2 * ge - 1          # rounds 0..ge-1 = task 0, rounds ge..2ge-2 = task 1
    proc_a = run_main(base + ["--method", "fedta_a3",
                              "--a3_lambda", "3e-7", "--a3_tau", "60",
                              "--a3_n_ref", "256",
                              "--run_name", "a3_smoke",
                              "--max_rounds", str(n_rounds_a)],
                      label="fedta_a3 short run")

    # ---- run B: plain fedta baseline (A3 off) ----
    run_main(base + ["--method", "fedta", "--run_name", "a3_smoke_ft",
                     "--max_rounds", str(n_rounds_a)],
             label="fedta baseline short run")

    # S1: baseline checkpoints carry no a3_state (schema invariance)
    ckpt_ft = torch_load_compat(run_ft / "checkpoints" / "latest.pth")
    for c in ckpt_ft["clients"]:
        assert "a3_state" not in c, (
            f"client {c['id']}: fedta baseline checkpoint must NOT carry "
            f"the a3_state key (schema invariance)")
    print("[PASS] S1: A3 off — fedta baseline runs and keeps its checkpoint "
          "schema untouched")

    # S2: task 0 inactive (empty bank in the task-0-end checkpoint)
    ckpt_t0 = torch_load_compat(run_a3 / "checkpoints" / "task_00_end.pth")
    for c in ckpt_t0["clients"]:
        a3 = c.get("a3_state")
        assert isinstance(a3, dict) and a3, "task-0 checkpoint missing a3_state"
        assert a3["task_id"] == 0 == c["task_id"]
        assert a3["ref_raw_indices"] is None
        assert a3["teacher_probs"] is None
    assert "task 0: A3 inactive" in proc_a.stdout
    print("[PASS] S2: task 0 — a3_state present but empty, A3 inactive")

    # S3: task 1 bank + compat KL > 0
    ckpt_a = torch_load_compat(run_a3 / "checkpoints" / "latest.pth")
    pool_size = None
    for c in ckpt_a["clients"]:
        assert c["task_id"] == 1, (
            f"client {c['id']}: expected mid-task-1 checkpoint")
        a3 = c.get("a3_state")
        assert isinstance(a3, dict) and a3, "task-1 checkpoint missing a3_state"
        assert a3["task_id"] == 1
        assert a3["ref_raw_indices"] and len(a3["ref_raw_indices"]) == 256
        probs = a3["teacher_probs"]
        assert probs is not None and probs.dim() == 2
        assert probs.shape[0] == 256
        pool_size = probs.shape[1]
        assert 0.0 <= float(a3["teacher_top1"]) <= 1.0
    kls = compat_kl_values(proc_a.stdout)
    assert kls, "no [A3] epoch log lines found"
    assert all(t == 1 for t, _ in kls), "compat KL logged outside task 1?"
    assert all(v > 0 for _, v in kls), f"non-positive compat KL: {kls}"
    assert "teacher routing bank ready" in proc_a.stdout
    assert "feature gradient detached: True" in proc_a.stdout
    print(f"[PASS] S3: task 1 bank ready (256 refs, teacher probs "
          f"[256, {pool_size}]), raw compat KL > 0 on all "
          f"{len(kls)} logged epochs")

    # ---- S4 + S5: in-process gradient isolation & sampling convention ----
    from main import build_server
    from runtime_utils import clone_prompt_template
    import argparse as _ap
    from config.datasets_delay import get_args_parser

    parser = _ap.ArgumentParser("test")
    sub = parser.add_subparsers(dest="subparser_name")
    get_args_parser(sub.add_parser("datasets_delay"))
    ip_args = parser.parse_args(
        ["datasets_delay",
         "--data_name", args.data_name, "--data_path", args.data_path,
         "--seed", str(args.seed), "--device", args.device,
         "--global_epoch", str(ge), "--task_num", str(args.task_num),
         "--local_epoch", str(args.local_epoch),
         "--method", "fedta_a3", "--a3_lambda", "3e-7",
         "--a3_tau", "60", "--a3_n_ref", "256"])
    server = build_server(ip_args)
    server.init_client()          # injects surrogate_data for fedta_a3

    # client 1 -> task-0 inactiveness check (in-process)
    c1 = server.clients[1]
    c1.prompts = clone_prompt_template(c1.vit)
    c1.task_id = 0
    c1._a3_init_task_bank(ip_args)
    assert c1._a3_teacher_probs is None and c1._a3_ref_raw is None

    # client 0 -> task-1 bank (teacher prompt = stored previous-gen prompt)
    c0 = server.clients[0]
    c0.prompts = clone_prompt_template(c0.vit)
    c0.task_id = 1
    c0._a3_init_task_bank(ip_args)
    assert c0._a3_teacher_probs is not None
    assert c0._a3_teacher_probs.shape == (256, pool_size)
    assert c0._a3_teacher_top1 is not None

    # S5: sampling convention == a3_offline_calibration
    surro = server.surrogate_data
    targets = np.array(surro.targets)
    classes = sorted({int(t) for t in surro.targets})
    pool_raw = np.concatenate([np.where(targets == c)[0] for c in classes])
    gen = torch.Generator()
    gen.manual_seed(9700 + 100 * 0 + 1)
    perm = torch.randperm(int(pool_raw.shape[0]), generator=gen)[:256]
    expected = [int(pool_raw[i]) for i in perm.tolist()]
    assert c0._a3_ref_raw == expected, (
        "reference bank does not match the offline-calibration sampling "
        "convention (seed 9700 + 100*client_id + task_id over the "
        "class-sorted canonical pool)")
    print("[PASS] S5: bank indices match the diagnostic sampling convention")

    # S4: gradient isolation — compat-only backward touches ONLY the key
    c0.model.zero_grad(set_to_none=True)
    c0.vit.zero_grad(set_to_none=True)
    c0.prompts.zero_grad(set_to_none=True)
    loss_kl = c0._a3_compat_loss(0)
    assert float(loss_kl) >= 0.0, "KL must be non-negative"
    loss_kl.backward()
    assert c0.model.key.grad is not None, "key received no compat gradient"
    assert float(c0.model.key.grad.abs().sum()) > 0, "key compat gradient is 0"
    assert c0.model.anchor_pool.grad is None, \
        "compat gradient leaked into anchor_pool"
    assert all(p.grad is None for p in c0.model.head.parameters()), \
        "compat gradient leaked into the classification head"
    assert all(p.grad is None for p in c0.vit.prompt.parameters()), \
        "compat gradient leaked into the ViT prompt module"
    assert all(p.grad is None for p in c0.prompts.parameters()), \
        "compat gradient leaked into the stored (teacher) prompt module"
    print("[PASS] S4: compat-only backward -> key.grad != 0; "
          "anchor / head / vit-prompt / stored-prompt grads all None")

    # S7: KL direction — numerical assertion KL(R_old || R_new). At the
    # teacher fixed point (unperturbed key, identical prompts) both KL
    # directions are ~0, and for WEAK perturbations they stay nearly equal
    # (KL is symmetric to second order), so the key is perturbed with
    # progressively stronger directional noise until the two directions
    # separate by >= 5%; only then is loss-vs-forward-KL meaningful.
    # (Row scaling alone is invisible to l2_normalize; row PERMUTATION is
    # also useless — it permutes r_new columns and leaves fwd ~= rev.)
    key_backup = c0.model.key.detach().clone()
    bs = c0.a3_ref_batch_size
    pert_gen = torch.Generator().manual_seed(20261015)
    noise = torch.randn(c0.model.key.shape, generator=pert_gen,
                        dtype=key_backup.dtype).to(key_backup.device)
    unit = float(key_backup.abs().mean())

    def _fwd_rev_kl():
        loss_val = float(c0._a3_compat_loss(0))
        with torch.no_grad():
            raw = c0._a3_ref_raw[:bs]
            x = c0._a3_fetch_ref_images(raw).to(c0.device)
            out = c0.original_model(x)
            cls_features = out['pre_logits']
            out = c0.vit(x, task_id=c0.task_id, cls_features=cls_features,
                         train=True)
            feat = out['feat'].to(c0.device)
            x_embed_norm = c0.model.l2_normalize(feat, dim=1)
            key_rows = c0.model.key.reshape(-1, c0.model.key_size)
            key_norm = c0.model.l2_normalize(key_rows, dim=1)
            sim = torch.matmul(
                x_embed_norm, key_norm.t().to(x_embed_norm.device))
            log_r_new = torch.log_softmax(c0.a3_tau * sim, dim=1)
            r_new = log_r_new.exp()
            r_old = c0._a3_teacher_probs[:bs].to(c0.device)
            log_r_old = r_old.clamp_min(1e-30).log()
            # KL(P||Q) = sum P * (log P - log Q), mean over the batch
            kl_fwd = (r_old * (log_r_old - log_r_new)).sum(dim=1).mean()
            kl_rev = (r_new * (r_new.clamp_min(1e-30).log() - log_r_old)
                      ).sum(dim=1).mean()
        return loss_val, float(kl_fwd), float(kl_rev)

    chosen = None
    for scale in (3.0, 10.0, 1.0):
        c0.model.key.data.copy_(key_backup)
        c0.model.key.data.add_(noise * (scale * unit))
        loss_val, fwd, rev = _fwd_rev_kl()
        if fwd > 1e-2 and abs(fwd - rev) > 0.05 * max(fwd, rev):
            chosen = (loss_val, fwd, rev)
            break
    c0.model.key.data.copy_(key_backup)
    assert chosen is not None, (
        f"no perturbation scale separated the KL directions "
        f"(last: loss={loss_val:.6f}, fwd={fwd:.6f}, rev={rev:.6f})")
    loss_val, fwd, rev = chosen
    tol = 1e-4 + 1e-3 * abs(fwd)
    assert abs(loss_val - fwd) < tol, (
        f"compat loss {loss_val:.6f} != forward KL {fwd:.6f} "
        f"(KL(R_old||R_new) direction is wrong?)")
    assert abs(loss_val - rev) > abs(loss_val - fwd) + 0.5 * abs(fwd - rev), (
        f"compat loss {loss_val:.6f} is closer to the REVERSE KL "
        f"{rev:.6f} than to the forward KL {fwd:.6f}")
    print(f"[PASS] S7: compat loss == KL(R_old||R_new) numerically "
          f"(loss={loss_val:.6f}, fwd={fwd:.6f}, rev={rev:.6f}; "
          f"the two directions are clearly distinguishable)")

    # S8: fetch immutability — images come from surro.data[raw_idx], never
    # from the mutable TrainData buffer that getTrainData() re-orders.
    from PIL import Image
    probe = c0._a3_ref_raw[:4]
    imgs_before = c0._a3_fetch_ref_images(probe)
    direct = torch.stack(
        [surro.trans(Image.fromarray(surro.data[i]).convert('RGB'))
         for i in probe])
    assert torch.equal(imgs_before, direct), (
        "_a3_fetch_ref_images does not read surro.data[raw_idx] "
        "(byte-inequal to a direct immutable-array read)")
    # force the TrainData reorder kd_fusion_prompt performs every round
    surro.getTrainData(list(reversed(classes)))
    assert int(surro.TrainLabels[0]) == int(classes[-1]), (
        "getTrainData did not reorder TrainData — S8 would be vacuous")
    imgs_after = c0._a3_fetch_ref_images(probe)
    assert torch.equal(imgs_before, imgs_after), (
        "_a3_fetch_ref_images output changed after a TrainData reorder "
        "(the bank silently follows the mutable buffer)")
    print("[PASS] S8: fetch reads immutable surro.data[raw_idx]; unchanged "
          "after a getTrainData reorder of the mutable TrainData buffer")

    # ---- S6: mid-task resume must restore the identical bank ----
    ckpt_r2 = torch_load_compat(run_a3 / "checkpoints" /
                                f"round_{n_rounds_a - 1:04d}.pth")
    proc_r = run_main(base + ["--method", "fedta_a3",
                              "--a3_lambda", "3e-7", "--a3_tau", "60",
                              "--a3_n_ref", "256",
                              "--run_name", "a3_smoke", "--resume", "auto",
                              "--max_rounds", str(2 * ge)],
                      label="fedta_a3 mid-task resume run")
    assert "restored reference bank" in proc_r.stdout, (
        "resume did not restore the a3 reference bank from the checkpoint")
    ckpt_r3 = torch_load_compat(run_a3 / "checkpoints" / "latest.pth")
    assert ckpt_r3["completed_round"] == n_rounds_a
    for c2, c3 in zip(ckpt_r2["clients"], ckpt_r3["clients"]):
        assert c2["id"] == c3["id"]
        a2s, a3s = c2["a3_state"], c3["a3_state"]
        assert a3s["task_id"] == 1
        assert a2s["ref_raw_indices"] == a3s["ref_raw_indices"], (
            f"client {c2['id']}: resume re-sampled the reference bank!")
        assert torch.equal(a2s["teacher_probs"], a3s["teacher_probs"]), (
            f"client {c2['id']}: resume changed the teacher targets!")
    kls_r = compat_kl_values(proc_r.stdout)
    assert kls_r and all(v > 0 for _, v in kls_r)
    print("[PASS] S6: mid-task resume restored bit-identical teacher targets "
          "and the same reference indices (no re-sampling)")

    print("\nRESULT: PASS (S1-S8)")


if __name__ == "__main__":
    main()
