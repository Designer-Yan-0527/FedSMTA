"""Phase 6 A4: runtime gate tests S1-S8 (single script, single bootstrap
pair, minimal batches). Converts the implementation-review debate into
executable assertions; run this BEFORE any formal A4 experiment.

Requires a source run with --method fedta_a4 that has completed at least
task 0 (checkpoints/task_00_end.pth exists). The gate bootstraps that
task boundary and drives the A4 machinery directly; it NEVER writes into
the source run_dir (all state churn happens in memory; the S8 checkpoint
round-trip goes through a temp directory that is removed afterwards).

Gates:
  S1 protected-set identity: two independent protection initializations
     (A2-key semantics and A4 semantics share _a2_init_task_protection;
     both must produce EXACTLY the same slots + key snapshot — otherwise
     A4-vs-A2 stops being a hard/soft single-variable comparison)
  S2 zero-at-snapshot: right after task-start init, L_dir == 0 and its
     gradient == 0 (K == K_old there)
  S3 cosine equivalence: after perturbing only the protected rows, the
     implemented loss equals an INDEPENDENTLY-coded reference
     2*(1-cos) formula (no shared helpers — a bug must not verify itself)
  S4 gradient scope: backward(_a4_dir_loss()) touches ONLY the protected
     key rows: unprotected key rows, anchor_pool, head AND every
     trainable ViT/prompt parameter (client.vit — prompts do not live
     in the Tail_Anchor model) all zero/None
  S5 tangential gradient: per protected row |<g_dir, k_hat>|/|g_dir| —
     the normalization Jacobian (I - k_hat k_hat^T)/||k|| must project
     g_dir onto the tangent space (max radial leakage < 1e-5; zero-grad
     rows reported separately, never clamped away)
  S6 soft-not-hard preservation: one REAL optimizer step on
     base + lambda*L_dir moves BOTH the protected key rows (no hidden
     post-step restore) and the anchor (plasticity preserved)
  S7 task-transition equivalence: the OFFICIAL server-ordered transition
     (update_data; protection init runs inside it on the task switch) vs
     THE ACTUAL probe stage-1 (a4_lambda_probe.prepare_probe_task,
     imported and invoked — not a re-implementation), compared per
     client on train/test split indices, protected slots and key
     snapshot by VALUE (closes the RNG-ordering question and would catch
     any future regression of the probe's ordering)
  S8 A4 checkpoint round-trip equivalence: fixed-batch trajectory
       step1, step2 | save -> reload -> fresh-optimizer step3
     vs
       step1, step2 | step3 (continuous)
     must end bit-identical on key / anchor / slots / snapshot. This
     proves A4-state serialization/restore equivalence ONLY — NOT an
     end-to-end federated-round resume proof (prompt fusion, all-client
     aggregation etc. are deliberately out of scope). Batches
     are pre-fetched tensors fed to both paths, so this tests A4-state
     checkpointing only — DataLoader RNG restoration is deliberately NOT
     mixed in (official checkpoints are round-boundary: the per-round
     Adam optimizer is rebuilt on resume, which is exactly what the
     fresh-optimizer step3 replicates).

Exit code 0 only if all eight gates PASS. Results JSON carries the meta
gate (commit hash / run_dir / checkpoint / seed / task / client / torch
version / device) so a PASS file stays attributable to a commit.

Usage (from repo root, on the server):
  python diagnostics/a4_gate_tests.py \
      --run_dir output/cifar100/fedta_a4/<run_name>/seed_42 \
      --gate_task 1
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from diag_utils import load_args_from_run_dir, resolve_checkpoint
from a4_lambda_probe import phase2_forward_loss, prepare_probe_task
from runtime_utils import atomic_save_checkpoint

# FP32 tolerances (GPT round-3 spec): the snapshot-point zeros are exact
# in theory, so only tiny numerical slack is allowed; the S3 reference
# uses a different numerical path, so a relative tolerance is right.
TOL_S2_LOSS = 1e-12
TOL_S2_GRAD = 1e-10
TOL_S3_REL = 1e-5
TOL_S5_MAX_RADIAL = 1e-5
GATE_LAMBDA = 1.0  # construction-only value to drive optimizer steps

RESULTS = []


def gate(name, ok, detail):
    RESULTS.append({"gate": name, "pass": bool(ok), "detail": detail})
    pad = "." * max(1, 34 - len(name))
    print(f"{name} {pad} {'PASS' if ok else 'FAIL'}")
    for k, v in detail.items():
        print(f"    {k}: {v}")
    return ok


def build_gate_server(run_dir, checkpoint, device=None, data_path=None):
    """Bootstrap with the source run's OWN args (the source must already
    be a fedta_a4 run, so the checkpoint args match on resume — no
    method overriding, no validate_checkpoint_args surprises)."""
    from main import build_server
    args = load_args_from_run_dir(run_dir, data_path=data_path,
                                  device=device)
    if args.method != 'fedta_a4':
        raise SystemExit(
            f"a4_gate_tests requires a fedta_a4 source run, got "
            f"method={args.method!r} in {run_dir}/args.json (the S8 "
            f"checkpoint round-trip validates method/lambda consistency)")
    server = build_server(args, run_manager=None)
    server.init_client()
    if checkpoint is not None:
        server.load_checkpoint(resolve_checkpoint(run_dir, checkpoint))
    return args, server


def server_task_transition(server, run_args, gate_round):
    """All clients complete the task transition IN SERVER ORDER (the
    train_clients() order); update_data itself runs the A2/A4 protection
    init on the task switch."""
    for cid in range(server.client_num):
        server.clients[cid].update_data(round=gate_round, args=run_args)


def pick_client(server, gate_task, wanted):
    c = server.clients[wanted]
    if c.task_id != gate_task or c._a2_protected_slots is None:
        for c2 in server.clients:
            if c2.task_id == gate_task and c2._a2_protected_slots is not None:
                return c2
        raise SystemExit(f"no client has a protection set for task "
                         f"{gate_task} (gate_task must be >= 1)")
    return c


def fetch_fixed_batches(client, run_args, n):
    """Pre-fetch n fixed batches as device tensors (identical inputs for
    every path — RNG independence by construction)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(client.traindata, batch_size=16,
                        num_workers=run_args.num_workers, shuffle=True)
    it = iter(loader)
    batches = []
    for _ in range(n):
        try:
            input, target = next(it)
        except StopIteration:
            it = iter(loader)
            input, target = next(it)
        batches.append((input.to(client.device), target.long().to(client.device)))
    return batches


def a4_step(client, run_args, round_id, batch, optimizer):
    """One REAL A4 phase-2 step: official base loss + lambda * L_dir."""
    input, target = batch
    loss = phase2_forward_loss(client, run_args, round_id, input, target)
    loss = loss + client.a4_lambda * client._a4_dir_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main():
    p = argparse.ArgumentParser("A4 runtime gate tests S1-S8")
    p.add_argument("--run_dir", required=True,
                   help="fedta_a4 run directory (task boundary checkpoint "
                        "must exist)")
    p.add_argument("--gate_task", type=int, default=1,
                   help="task the gates run on (checkpoint default: "
                        "task_{gate_task-1:02d}_end.pth)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--client", type=int, default=0,
                   help="client used for S1-S6/S8 (falls back to any "
                        "client with a non-empty protection set)")
    p.add_argument("--out_json",
                   default="reports/a4_gate_results.json")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ckpt = args.checkpoint or (
        f"checkpoints/task_{args.gate_task - 1:02d}_end.pth")
    run_args, server = build_gate_server(
        args.run_dir, ckpt, device=args.device, data_path=args.data_path)
    gate_round = args.gate_task * run_args.global_epoch

    server_task_transition(server, run_args, gate_round)
    client = pick_client(server, args.gate_task, args.client)
    print(f"gate client = client {client.id} "
          f"(task {client.task_id}, "
          f"{client._a2_protected_slots.numel()} protected rows)")

    key = client.model.key
    slots = client._a2_protected_slots

    # ---------------- S1: protected-set identity (A2 vs A4 semantics) ----
    snap1 = client._a2_key_snapshot.clone()
    slots1 = slots.clone()
    # A2-key semantics: identical function, identical inputs (a2_arm is
    # 'key' for both arms); init is RNG-neutral so a second call is safe
    client._a2_protected_slots = None
    client._a2_key_snapshot = None
    client._a2_anchor_snapshot = None
    client.a2_arm = 'key'
    client._a2_init_task_protection(run_args)
    ok1 = (torch.equal(slots1, client._a2_protected_slots)
           and torch.equal(snap1, client._a2_key_snapshot))
    gate("S1 shared protection-path identity", ok1, {
        "n_protected": int(slots1.numel()),
        "slots_equal": bool(torch.equal(slots1, client._a2_protected_slots)),
        "snapshot_equal": bool(torch.equal(snap1, client._a2_key_snapshot)),
        "note": "A2-key and A4 intentionally share the same protection "
                "initializer (_a2_init_task_protection, a2_arm='key'); "
                "this gate verifies exact deterministic re-initialization "
                "on the shared path, NOT a cross-method equivalence "
                "between two independent implementations",
    })

    # ---------------- S2: zero at the snapshot point ---------------------
    client.model.zero_grad()
    loss0 = client._a4_dir_loss()
    l0 = float(loss0.detach())
    loss0.backward()
    g0 = key.grad[slots].detach()
    gn0 = float(g0.norm())
    client.model.zero_grad()
    gate("S2 zero-at-snapshot", l0 <= TOL_S2_LOSS and gn0 <= TOL_S2_GRAD, {
        "dir_loss": l0, "grad_norm": gn0,
        "thresholds": [TOL_S2_LOSS, TOL_S2_GRAD],
    })

    # ---------------- S3: independent 2(1-cos) reference -----------------
    with torch.no_grad():
        key.data[slots] += torch.randn_like(key.data[slots]) * 1e-3
    loss_impl = float(client._a4_dir_loss().detach())
    # independent reference: NO shared helper with the implementation
    k = key.data[slots].double()
    k_old = client._a2_key_snapshot.to(k.dtype)
    k_hat = F.normalize(k, dim=1, eps=1e-12)
    k_old_hat = F.normalize(k_old, dim=1, eps=1e-12)
    loss_ref = float((2.0 * (1.0 - (k_hat * k_old_hat).sum(dim=1))).mean())
    rel = abs(loss_impl - loss_ref) / max(abs(loss_ref), 1e-30)
    gate("S3 cosine equivalence", rel < TOL_S3_REL, {
        "loss_impl": loss_impl, "loss_reference": loss_ref,
        "abs_diff": abs(loss_impl - loss_ref), "rel_diff": rel,
    })

    # ---------------- S4: gradient scope ---------------------------------
    client.model.zero_grad()
    client.vit.zero_grad()
    client._a4_dir_loss().backward()
    g = key.grad.detach()
    prot_max = float(g[slots].abs().max())
    mask = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
    mask[slots] = False
    unprot_max = float(g[mask].abs().max()) if mask.any() else 0.0
    anchor_max = 0.0
    if client.model.anchor_pool.grad is not None:
        anchor_max = float(client.model.anchor_pool.grad.abs().max())
    non_key_max = 0.0
    for pn, pm in client.model.named_parameters():
        if pm.grad is not None and not pn.startswith("key"):
            non_key_max = max(non_key_max, float(pm.grad.abs().max()))
    # prompts / ViT live in client.vit, NOT in the Tail_Anchor model —
    # A4 must never touch them; check every trainable ViT parameter
    vit_max = 0.0
    n_vit_trainable = 0
    for pm in client.vit.parameters():
        if pm.requires_grad:
            n_vit_trainable += 1
            if pm.grad is not None:
                vit_max = max(vit_max, float(pm.grad.abs().max()))
    gate("S4 gradient scope",
         prot_max > 0 and unprot_max == 0 and anchor_max == 0
         and non_key_max == 0 and vit_max == 0, {
             "protected_key_grad_max": prot_max,
             "unprotected_key_grad_max": unprot_max,
             "anchor_grad_max": anchor_max,
             "non_key_param_grad_max": non_key_max,
             "vit_prompt_grad_max": vit_max,
             "n_vit_trainable_params": n_vit_trainable,
         })

    # ---------------- S5: tangential gradient ----------------------------
    k_now = key.data[slots]
    k_hat_now = client.model.l2_normalize(k_now, dim=1)
    g_dir = g[slots]
    gd = g_dir.norm(dim=1)
    valid = gd > 1e-12
    n_zero = int((~valid).sum())
    radial = ((g_dir * k_hat_now).sum(dim=1).abs()
              / (gd + 1e-30))[valid]
    max_radial = float(radial.max()) if radial.numel() else float("nan")
    med_radial = (float(radial.median()) if radial.numel()
                  else float("nan"))
    gate("S5 tangent gradient",
         bool(valid.any()) and max_radial < TOL_S5_MAX_RADIAL, {
             "max_radial_leakage": max_radial,
             "median_radial_leakage": med_radial,
             "n_zero_dir_rows": n_zero,
             "n_valid_rows": int(valid.sum()),
             "threshold": TOL_S5_MAX_RADIAL,
         })

    # ---------------- S6: soft-not-hard preservation ---------------------
    batches = fetch_fixed_batches(client, run_args, 3)
    k_before = key.data.clone()
    a_before = client.model.anchor_pool.data.clone()
    opt = torch.optim.Adam(client.model.parameters(), lr=client.lr,
                           weight_decay=1e-03)
    a4_step(client, run_args, gate_round, batches[0], opt)
    k_after = key.data.clone()
    a_after = client.model.anchor_pool.data.clone()
    prot_delta = float((k_after[slots] - k_before[slots]).norm())
    unprot_delta = float((k_after[mask] - k_before[mask]).norm())
    anchor_delta = float((a_after - a_before).norm())
    gate("S6 soft-not-hard preservation",
         prot_delta > 0 and anchor_delta > 0, {
             "protected_key_delta_norm": prot_delta,
             "unprotected_key_delta_norm": unprot_delta,
             "anchor_delta_norm": anchor_delta,
         })

    # ---------------- S7: task-transition equivalence --------------------
    # Path A = the OFFICIAL transition (server-ordered update_data; the
    #          protection init runs inside update_data on the task switch
    #          — exactly what Server_DF.train_clients() triggers)
    # Path B = THE ACTUAL probe stage-1 function (prepare_probe_task,
    #          imported from a4_lambda_probe — not a re-implementation),
    #          so a future regression of the probe's ordering would be
    #          caught here, not silently pass
    _, server_b = build_gate_server(
        args.run_dir, ckpt, device=args.device, data_path=args.data_path)
    prepare_probe_task(server_b, run_args, gate_round, args.gate_task)
    detail = {"path_b": "a4_lambda_probe.prepare_probe_task (actual "
                        "probe stage-1)",
              "clients_checked": 0, "all_equal": True}
    for cid in range(server.client_num):
        ca, cb = server.clients[cid], server_b.clients[cid]
        if ca._a2_protected_slots is None and cb._a2_protected_slots is None:
            continue
        detail["clients_checked"] += 1
        train_eq = (ca.data_split_indices[args.gate_task]["train"]
                    == cb.data_split_indices[args.gate_task]["train"])
        test_eq = (ca.data_split_indices[args.gate_task]["test"]
                   == cb.data_split_indices[args.gate_task]["test"])
        slots_eq = (ca._a2_protected_slots is None
                    and cb._a2_protected_slots is None) or (
            ca._a2_protected_slots is not None
            and cb._a2_protected_slots is not None
            and torch.equal(ca._a2_protected_slots,
                            cb._a2_protected_slots))
        snap_eq = (ca._a2_key_snapshot is None
                   and cb._a2_key_snapshot is None) or (
            ca._a2_key_snapshot is not None
            and cb._a2_key_snapshot is not None
            and torch.equal(ca._a2_key_snapshot,
                            cb._a2_key_snapshot))
        eq = train_eq and test_eq and slots_eq and snap_eq
        detail[f"client{cid}"] = (f"split={train_eq and test_eq} "
                                  f"slots={slots_eq} snapshot={snap_eq}")
        detail["all_equal"] = detail["all_equal"] and eq
    gate("S7 task-transition equivalence",
         detail["all_equal"] and detail["clients_checked"] > 0, detail)

    # ---------------- S8: resume equivalence -----------------------------
    # server_b is still at a clean task-start state (only update_data ran)
    c8 = pick_client(server_b, args.gate_task, args.client)
    if c8._a2_protected_slots is None:
        raise SystemExit("S8: no protection set available")
    c8.original_model.eval()
    if c8.prompts is not None:
        c8.vit.load_prompts(c8.prompts)
    c8.model.to(c8.device)
    c8.vit.to(c8.device)
    batches8 = fetch_fixed_batches(c8, run_args, 3)

    # shared prefix: two steps on one optimizer simulate completing the
    # FIRST ROUND of gate_task; save a round-boundary checkpoint of the
    # WHOLE server to a temp dir. round_id = gate_round (completed round
    # belongs to gate_task, matching the state being saved — a checkpoint
    # labeled gate_round-1 would claim the previous task's last round
    # while holding gate_task state, a state that never exists in
    # production). server.task_id is set to match the simulated round
    # completion (train_clients updates it when the round's task begins).
    opt8 = torch.optim.Adam(c8.model.parameters(), lr=c8.lr,
                            weight_decay=1e-03)
    a4_step(c8, run_args, gate_round, batches8[0], opt8)
    a4_step(c8, run_args, gate_round, batches8[1], opt8)
    server_b.task_id = args.gate_task
    tmp_dir = tempfile.mkdtemp(prefix="a4_gate_s8_")
    tmp_ckpt = str(Path(tmp_dir) / "s8_round_boundary.pth")
    atomic_save_checkpoint(server_b.build_checkpoint(gate_round),
                           tmp_ckpt)

    # Path A: continuous — fresh per-round optimizer (official resume
    # semantics: train() rebuilds Adam every round; requires
    # global_epoch >= 2 so gate_round+1 is still inside gate_task),
    # step3 in memory
    optA = torch.optim.Adam(c8.model.parameters(), lr=c8.lr,
                            weight_decay=1e-03)
    a4_step(c8, run_args, gate_round, batches8[2], optA)
    k_a = c8.model.key.data.clone()
    a_a = c8.model.anchor_pool.data.clone()
    s_a = c8._a2_protected_slots.clone()
    snap_a = c8._a2_key_snapshot.clone()

    # Path B: reload the temp checkpoint into a FRESH bootstrap and take
    # the same step3 with a fresh optimizer
    _, server_d = build_gate_server(
        args.run_dir, tmp_ckpt, device=args.device, data_path=args.data_path)
    c8d = server_d.clients[c8.id]
    c8d.original_model.eval()
    if c8d.prompts is not None:
        c8d.vit.load_prompts(c8d.prompts)
    c8d.model.to(c8d.device)
    c8d.vit.to(c8d.device)
    optB = torch.optim.Adam(c8d.model.parameters(), lr=c8d.lr,
                            weight_decay=1e-03)
    a4_step(c8d, run_args, gate_round, batches8[2], optB)
    k_b = c8d.model.key.data.clone()
    a_b = c8d.model.anchor_pool.data.clone()

    key_eq = torch.equal(k_a, k_b)
    anchor_eq = torch.equal(a_a, a_b)
    slots_eq = torch.equal(s_a, c8d._a2_protected_slots)
    snap_eq = torch.equal(snap_a, c8d._a2_key_snapshot)
    max_diff = float((k_a - k_b).abs().max())
    gate("S8 A4 checkpoint round-trip equivalence",
         key_eq and anchor_eq and slots_eq and snap_eq, {
             "key_equal": bool(key_eq), "anchor_equal": bool(anchor_eq),
             "slots_equal": bool(slots_eq),
             "snapshot_equal": bool(snap_eq),
             "key_max_abs_diff": max_diff,
             "scope": "A4-state serialization/restore equivalence only; "
                      "not an end-to-end federated-round resume proof",
         })

    # cleanup temp checkpoint (never touches the source run_dir)
    try:
        Path(tmp_ckpt).unlink()
        Path(tmp_dir).rmdir()
    except OSError:
        pass

    # ---------------- meta + summary -------------------------------------
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    n_pass = sum(1 for r in RESULTS if r["pass"])
    meta = {
        "commit": commit,
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt),
        "seed": getattr(run_args, "seed", None),
        "gate_task": args.gate_task,
        "gate_round": gate_round,
        "client": client.id,
        "method": run_args.method,
        "a4_lambda_at_source": getattr(run_args, "a4_lambda", None),
        "gate_lambda": GATE_LAMBDA,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "tolerances": {"s2_loss": TOL_S2_LOSS, "s2_grad": TOL_S2_GRAD,
                       "s3_rel": TOL_S3_REL, "s5_max_radial": TOL_S5_MAX_RADIAL},
    }
    out = {"meta": meta, "gates": RESULTS,
           "overall": f"{n_pass}/{len(RESULTS)}",
           "verdict": "PASS" if n_pass == len(RESULTS) else "FAIL"}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nA4 GATE: {out['verdict']} ({n_pass}/{len(RESULTS)})")
    print(f"results: {out_path}")
    print(f"commit: {commit}")
    raise SystemExit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
