"""Phase 4 D0 (revised): routing-structure identification for Tail_Anchor.

D0 is a DESCRIPTIVE topology-identification experiment, not a verdict
machine. Its outputs decide whether D1 uses single-slot attribution,
weighted multi-slot attribution, or full routing-distribution analysis.

Measures, per (client, task), on ONE fixed augmented view of the task's
train split (materialized ONCE so both generation states see bit-identical
inputs — augmentation RNG jitter fully isolated):

  1. class-side concentration: diag_share, modal_share, routing entropy;
  2. slot-side specificity: modal-slot purity, #classes sharing the slot
     (within-task AND cross-task — a task-3 class stealing a task-0
     class's slot is the core collision shape and invisible to
     within-task purity);
  3. paired cross-generation routing stability (t_end state vs final
     deployment state, same cached features), decomposed routing-2x2:
     R_oo/R_oc isolates KEY drift, R_oo/R_co isolates FEATURE (prompt)
     drift — same coordinates as the accuracy 2x2 of compatibility_drift.py,
     bridging D0 directly to the KA/P channel decomposition.

Routing/feature path stays faithful to official evaluate() semantics:
vit forward with train=True (batchwise_prompt path), matching
compatibility_drift.py / analyze_class_retention.py so D0 routing
distributions are directly comparable to the validated 2x2 diagnostics.
Determinism comes from fixed seeds + cached inputs, not from switching
to eval mode (which would change prompt selection semantics).

Observer-only: no training code touched; RNG snapshot/restored; prompt
restored in finally; empty results raise.

Usage (repo root, on the server):
  python diagnostics/verify_slot_binding.py \
      --run_dir output/cifar100/fedta/<run>/seed_42 \
      --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
      --out_csv diagnostics_output/slot_binding.csv \
      --out_matrix_npz diagnostics_output/slot_binding_matrices.npz \
      --out_json diagnostics_output/slot_binding_summary.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server
from runtime_utils import (load_checkpoint_file, clone_prompt_template,
                           rng_state_dict, restore_rng_state)
from slot_collision_diagnostic import l2_normalize, get_train_split, resolve


def parse_args():
    p = argparse.ArgumentParser("Phase-4 D0 slot-binding topology verification")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--task_ckpts", required=True,
                   help="comma-separated task_end checkpoints in task order")
    p.add_argument("--out_csv", default="diagnostics_output/slot_binding.csv")
    p.add_argument("--out_matrix_npz",
                   default="diagnostics_output/slot_binding_matrices.npz")
    p.add_argument("--out_json",
                   default="diagnostics_output/slot_binding_summary.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def capture_fixed_view(client, task_t, batch_size):
    """Materialize ONE augmented view of task t's train split (fixed seed).
    Both generation states route the SAME cached inputs, so routing
    differences are attributable to model state ONLY."""
    split_idx = get_train_split(client, task_t)
    dataset = Subset(client.train_data[task_t], split_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    inputs, targets = [], []
    for x, y in loader:
        inputs.append(x)
        targets.append(y)
    return torch.cat(inputs), torch.cat(targets)


def extract_features_on(client, inputs, task_t, prompt_module, batch_size):
    """vit features of the CACHED inputs under the given prompt state.
    Official evaluate() semantics: train=True forward path (batchwise
    prompt selection); deterministic because inputs are fixed and the
    caller seeds beforehand."""
    client.vit.load_prompts(prompt_module)
    client.vit.to(client.device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            inp = inputs[i:i + batch_size].to(client.device,
                                              non_blocking=True)
            output = client.original_model(inp)
            pre_logits = output['pre_logits'].requires_grad_(False)
            vit_out = client.vit(inp, task_id=task_t,
                                 cls_features=pre_logits, train=True)
            feats.append(vit_out['feat'].detach().cpu())
    return torch.cat(feats, dim=0)


def route_features(feat_tensor, key_tensor):
    """Deterministic routing replicating Tail_Anchor.forward
    (l2_normalize -> matmul -> topk(k=1))."""
    key_norm = l2_normalize(key_tensor, dim=1)
    x_norm = l2_normalize(feat_tensor, dim=1)
    sim = torch.matmul(x_norm, key_norm.t())
    return torch.topk(sim, k=1)[1].squeeze(1)


def entropy_bits(prob_dist):
    p = prob_dist[prob_dist > 0]
    return float(-torch.sum(p * torch.log2(p)).item()) if len(p) > 0 else 0.0


def main():
    args = parse_args()
    ckpt_paths = [resolve(args.run_dir, p) for p in args.task_ckpts.split(",")]
    n_bounds = len(ckpt_paths)

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=ckpt_paths[-1],
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes

    print("loading boundary checkpoints ...", flush=True)
    bounds = []
    for path in ckpt_paths:
        ckpt = load_checkpoint_file(path)
        per_client = {}
        for client in server.clients:
            st = ckpt["clients"][client.id]
            per_client[client.id] = {
                "prompts_state": st["prompts"],
                "key": st["tail_anchor_model"]["key"].detach().cpu().clone(),
            }
        bounds.append(per_client)
        del ckpt

    rng_snapshot = rng_state_dict()
    rows, matrices = [], {}
    summary_extra = {}
    try:
        for client in server.clients:
            cid = client.id
            cur_prompt = client.prompts
            try:
                pool_size = bounds[0][cid]["key"].shape[0]
                prompt_mods = {}
                for b in range(n_bounds):
                    pm = clone_prompt_template(client.vit)
                    pm.load_state_dict(bounds[b][cid]["prompts_state"])
                    prompt_mods[b] = pm
                final_prompt_mod = prompt_mods[n_bounds - 1]
                key_final = bounds[n_bounds - 1][cid]["key"]

                my_tasks = [t for t in range(len(client.train_data))
                            if client.heads[t] is not None]

                # cross-task binding matrix (client's full class sequence)
                B_client = torch.zeros(nb_classes, pool_size, dtype=torch.long)

                for t in my_tasks:
                    torch.manual_seed(9700 + 100 * cid + t)
                    inputs, targets = capture_fixed_view(
                        client, t, args.batch_size)

                    torch.manual_seed(9700 + 100 * cid + t + 50)
                    feats_t = extract_features_on(
                        client, inputs, t, prompt_mods[t], args.batch_size)
                    slots_t = route_features(
                        feats_t, bounds[t][cid]["key"]).cpu()

                    torch.manual_seed(9700 + 100 * cid + t + 50)
                    feats_f = extract_features_on(
                        client, inputs, t, final_prompt_mod, args.batch_size)
                    slots_f = route_features(feats_f, key_final).cpu()

                    # routing 2x2 (same coordinates as the accuracy 2x2 in
                    # compatibility_drift.py — bridges D0 to the KA/P
                    # channel decomposition):
                    #   R_oo = old features + old key      (= slots_t)
                    #   R_oc = old features + current key  (key drift)
                    #   R_co = current features + old key  (feature drift)
                    #   R_cc = current features + current key (= slots_f)
                    slots_oc = route_features(feats_t, key_final).cpu()
                    slots_co = route_features(feats_f,
                                             bounds[t][cid]["key"]).cpu()

                    print(f"client {cid} task {t}: routed "
                          f"({len(slots_t)} fixed samples)", flush=True)

                    # within-task binding matrix B (classes x pool)
                    B = torch.zeros(nb_classes, pool_size, dtype=torch.long)
                    for y, j in zip(targets.tolist(), slots_t.tolist()):
                        B[y, j] += 1
                    B_client += B
                    matrices[f"client_{cid}_task_{t}"] = B.numpy()

                    # LANDING distributions (D1-lite support-misalignment
                    # fix): where old-task features actually go under the
                    # CURRENT key (R_oc) and under the old key with
                    # current features (R_co) — damage happens at the
                    # landing slots, not the old support.
                    for tag, sl in [("_oc", slots_oc), ("_co", slots_co)]:
                        Bx = torch.zeros(nb_classes, pool_size,
                                         dtype=torch.long)
                        for y, j in zip(targets.tolist(), sl.tolist()):
                            Bx[y, j] += 1
                        matrices[f"client_{cid}_task_{t}{tag}"] = Bx.numpy()

                    a_key = (slots_t == slots_oc)    # R_oo vs R_oc
                    a_feat = (slots_t == slots_co)   # R_oo vs R_co
                    a_total = (slots_t == slots_f)   # R_oo vs R_cc
                    a_oc_cc = (slots_oc == slots_f)  # feat drift on new key
                    a_co_cc = (slots_co == slots_f)  # key drift on new feats
                    task_classes = [int(c) for c in client.class_mask[t]]
                    for c in task_classes:
                        m = (targets == c)
                        n_c = int(m.sum())
                        if n_c == 0:
                            continue
                        counts = B[c].float()
                        prob = counts / counts.sum()
                        modal_slot = int(torch.argmax(counts).item())
                        modal_share = float(prob[modal_slot].item())
                        diag_share = (float(prob[c].item())
                                      if c < pool_size else 0.0)
                        # within-task specificity of the modal slot
                        col_t = B[:, modal_slot]
                        purity_task = float(col_t.max() / col_t.sum()) \
                            if col_t.sum() > 0 else 0.0
                        # cross-task specificity (client's whole sequence)
                        col_c = B_client[:, modal_slot]
                        purity_client = float(col_c.max() / col_c.sum()) \
                            if col_c.sum() > 0 else 0.0
                        rows.append({
                            "client_id": cid,
                            "task": t,
                            "class_id": c,
                            "n_train": n_c,
                            "diag_share": diag_share,
                            "modal_slot": modal_slot,
                            "modal_share": modal_share,
                            "entropy_bits": entropy_bits(prob),
                            "modal_slot_purity_task": purity_task,
                            "modal_slot_purity_client": purity_client,
                            "modal_slot_sharing_task":
                                int((col_t > 0).sum().item()),
                            "modal_slot_sharing_client":
                                int((col_c > 0).sum().item()),
                            "stab_key_oo_oc": float(
                                a_key[m].float().mean().item()),
                            "stab_feat_oo_co": float(
                                a_feat[m].float().mean().item()),
                            "stab_oc_cc": float(
                                a_oc_cc[m].float().mean().item()),
                            "stab_co_cc": float(
                                a_co_cc[m].float().mean().item()),
                            "routing_stability": float(
                                a_total[m].float().mean().item()),
                            "final_diag_share": float(
                                (slots_f[m] == c).float().mean().item())
                            if c < pool_size else 0.0,
                        })

                matrices[f"client_{cid}_all"] = B_client.numpy()

                # top cross-task contended slots for this client
                totals = B_client.sum(dim=0)
                users = (B_client > 0).sum(dim=0)
                contended = [(int(j), int(users[j]), int(totals[j]))
                             for j in range(pool_size) if users[j] >= 2]
                contended.sort(key=lambda x: (-x[1], -x[2]))
                summary_extra[cid] = contended[:10]
            finally:
                if cur_prompt is not None:
                    client.vit.load_prompts(cur_prompt)
    finally:
        restore_rng_state(rng_snapshot)

    if not rows:
        raise RuntimeError("no rows collected — nothing to write")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(args.out_matrix_npz, **matrices)
    print(f"wrote {len(rows)} class rows -> {out_csv}")
    print(f"saved {len(matrices)} binding matrices -> {args.out_matrix_npz}")

    # ---------- descriptive topology audit (no mechanistic verdict) ----------
    import pandas as pd
    df = pd.DataFrame(rows)
    w_all = df.n_train
    def wmean(col):
        return float((df[col] * w_all).sum() / w_all.sum())

    summary = {
        "n_evaluated_classes": len(df),
        "weighted": {col: wmean(col) for col in
                     ["diag_share", "modal_share", "entropy_bits",
                      "modal_slot_purity_task", "modal_slot_purity_client",
                      "stab_key_oo_oc", "stab_feat_oo_co",
                      "stab_oc_cc", "stab_co_cc",
                      "routing_stability", "final_diag_share"]},
        "unweighted": {col: float(df[col].mean()) for col in
                       ["diag_share", "modal_share",
                        "modal_slot_purity_client", "routing_stability"]},
        "classes_with_exclusive_slot":
            int(((df.modal_share > 0.8) &
                 (df.modal_slot_purity_client > 0.8)).sum()),
        "classes_in_shared_slot_cross_task":
            int((df.modal_slot_sharing_client > 1).sum()),
        "top_contended_slots_per_client": summary_extra,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== topology audit summary (descriptive) =====")
    print(f"classes evaluated: {len(df)}")
    print(f"weighted diag_share={wmean('diag_share'):.3f}  "
          f"modal_share={wmean('modal_share'):.3f}  "
          f"entropy_bits={wmean('entropy_bits'):.2f}")
    print(f"weighted modal purity: task={wmean('modal_slot_purity_task'):.3f}"
          f"  client(cross-task)={wmean('modal_slot_purity_client'):.3f}")
    print(f"exclusive slots (modal>0.8 & cross-purity>0.8): "
          f"{summary['classes_with_exclusive_slot']}/{len(df)}")
    print(f"classes whose modal slot is shared cross-task: "
          f"{summary['classes_in_shared_slot_cross_task']}/{len(df)}")

    print("\n===== routing 2x2 decomposition (bridges to accuracy 2x2) =====")
    print(f"key drift    stab(R_oo,R_oc)={wmean('stab_key_oo_oc'):.3f}"
          f"  (1-agree = key-channel routing flip)")
    print(f"feat drift   stab(R_oo,R_co)={wmean('stab_feat_oo_co'):.3f}"
          f"  (1-agree = feature/prompt-channel routing flip)")
    print(f"cross terms  stab(R_oc,R_cc)={wmean('stab_oc_cc'):.3f}"
          f"  stab(R_co,R_cc)={wmean('stab_co_cc'):.3f}")
    print(f"total        stab(R_oo,R_cc)={wmean('routing_stability'):.3f}")
    d_key = 1.0 - wmean('stab_key_oo_oc')
    d_feat = 1.0 - wmean('stab_feat_oo_co')
    d_tot = 1.0 - wmean('routing_stability')
    if d_key + d_feat > d_tot:
        label = "overlap/redundant (both channels flip same samples)"
    else:
        label = "synergy (joint effect exceeds sum)"
    print(f"non-additivity: key={d_key:.3f} + feat={d_feat:.3f} "
          f"vs total={d_tot:.3f}  {label} "
          f"| gap {abs(d_key + d_feat - d_tot):.3f}")

    print("\n===== per (client, task) table (train-weighted) =====")
    for (cid, t), sub in df.groupby(["client_id", "task"]):
        w = sub.n_train
        def wm(col):
            return float((sub[col] * w).sum() / w.sum())
        print(f"c{cid} t{t}: diag={wm('diag_share'):.2f}  "
              f"modal={wm('modal_share'):.2f}  "
              f"purity_t={wm('modal_slot_purity_task'):.2f}  "
              f"purity_x={wm('modal_slot_purity_client'):.2f}  "
              f"stabK={wm('stab_key_oo_oc'):.2f}  "
              f"stabF={wm('stab_feat_oo_co'):.2f}  "
              f"stab={wm('routing_stability'):.2f}")

    print("\n===== top cross-task contended slots (client, task-agnostic) =====")
    for cid, lst in summary_extra.items():
        tops = ", ".join(f"slot{j}(u{u},h{h})" for j, u, h in lst[:5])
        print(f"c{cid}: {tops}")
    print("\nNOTE: no D1 verdict printed. Decide D1 attribution form "
          "(single-slot / weighted multi-slot / full distribution) after "
          "inspecting slot_binding.csv, the matrices npz and the summary "
          "json.")


if __name__ == "__main__":
    main()
