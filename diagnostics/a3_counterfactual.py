"""Phase 5 A3 counterfactual decomposition (eval-only, zero training).

Why: A3-1e-5 locks the surrogate-bank routing KL (compat_kl ~ 3.7, flat
over 30 epochs) yet old-task retention stays at baseline level.  Two
questions decide the next chapter:

Q1 (routing transfer): does the bank routing consistency ALSO hold on
    OLD-TASK PRIVATE test samples?  NOTE: cur_ckpt is saved AFTER
    server prompt fusion, so the bank KL measured here is the
    POST-FUSION deployed consistency -- deliberately different from
    the local phase-2 compat_kl in the training log (pre-fusion
    local prompt).  Three-branch reading:
      bank KL high  -> fusion breaks what A3 learned (injection-point
                       problem, not a mechanism failure)
      bank low / private high -> surrogate reference mismatch
      both low      -> routing consistency itself insufficient
    Metrics per client: KL(R_old||R_new), top-1 flip rate, routing
    agreement, on (a) the private task-t test set and (b) the stored A3
    reference bank.  Private metrics are split by EMERGENT-SUPPORT
    rows (rows that task-t train features top-1 route to under the old
    generation -- the A2 protection-set definition) vs the rest.

Q2 (2x2 counterfactual): {old,new} prompt x {old,new} key+anchor
    evaluated on the old-task test set, A1/compatibility_drift protocol
    (current head, strict combo order, seeded eval).  Isolates prompt
    drift vs Tail_Anchor-side (key/anchor) drift vs their interaction.
      old = task_XX_end.pth generation (the task that learned t)
      new = --cur_ckpt generation (e.g. round_0008.pth mid-task-1)

Usage (repo root, on the machine holding the A3 run):
  python diagnostics/a3_counterfactual.py \
      --run_dir output/cifar100/fedta_a3/a3_lam1e-5/seed_42 \
      --cur_ckpt checkpoints/round_0008.pth \
      --old_ckpt checkpoints/task_00_end.pth \
      --task 0
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from diag_utils import bootstrap_server
from runtime_utils import (clone_prompt_template, load_checkpoint_file,
                           rng_state_dict, restore_rng_state)


def parse_args():
    p = argparse.ArgumentParser("A3 counterfactual decomposition")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--cur_ckpt", default="checkpoints/latest.pth",
                   help="new generation checkpoint (default latest.pth)")
    p.add_argument("--old_ckpt", default=None,
                   help="old generation checkpoint (default "
                        "checkpoints/task_{t:02d}_end.pth)")
    p.add_argument("--task", type=int, default=0,
                   help="old task to analyze (default 0)")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def resolve(run_dir, ckpt):
    path = Path(ckpt)
    if not path.is_absolute():
        path = Path(run_dir) / ckpt
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return str(path.resolve())


def routing_probs(client, feat, key):
    """R = softmax(tau * sim(z, K)) exactly as Tail_Anchor.forward /
    _a3_build_teacher: official l2-normalization, key rows as-is."""
    tau = float(getattr(client, "a3_tau", 60.0))
    x_norm = client.model.l2_normalize(feat, dim=1)
    key_rows = key.reshape(-1, client.model.key_size)
    key_norm = client.model.l2_normalize(key_rows, dim=1)
    sim = torch.matmul(x_norm, key_norm.t().to(x_norm.device))
    return torch.softmax(tau * sim, dim=1)


@torch.no_grad()
def extract_feats(client, prompt, images, batch_size=8):
    """Features under the given prompt module, mirroring evaluate():
    original_model -> pre_logits -> vit(train=True) -> feat."""
    client.vit.load_prompts(prompt)
    feats = []
    for i in range(0, images.shape[0], batch_size):
        x = images[i:i + batch_size].to(client.device, non_blocking=True)
        output = client.original_model(x)
        cls_features = output['pre_logits']
        output = client.vit(x, task_id=client.task_id,
                            cls_features=cls_features, train=True)
        feats.append(output['feat'].to(client.device))
    return torch.cat(feats, dim=0)


def kl_rows(p, q, eps=1e-12):
    """KL(P||Q) per row (both are probability tensors [N, K])."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1)


def routing_metrics(r_old, r_new, in_support=None):
    """KL / flip / agreement overall (+ by support split if given)."""
    top_old = r_old.argmax(dim=1)
    top_new = r_new.argmax(dim=1)
    flip = (top_old != top_new)
    kl = kl_rows(r_old, r_new)
    out = []

    def block(name, mask):
        n = int(mask.sum())
        if n == 0:
            return
        out.append({
            "split": name, "n_samples": n,
            "kl_mean": float(kl[mask].mean()),
            "flip_rate": float(flip[mask].float().mean()),
            "agreement": float((~flip[mask]).float().mean()),
            "old_top1_conf": float(r_old[mask].max(dim=1).values.mean()),
        })

    block("all", torch.ones_like(flip))
    if in_support is not None:
        block("in_support", in_support)
        block("out_support", ~in_support)
    return out


def eval_cell(client, task, nb_classes, ptag, prompt, ktag, ka, rows):
    """A1/compatibility_drift strict combo order:
    load prompt -> load KA -> set seed -> evaluate."""
    client.vit.load_prompts(prompt)
    client.model.load_state_dict(ka, strict=False)  # head untouched
    torch.manual_seed(9700 + 100 * client.id + task)
    acc = client.evaluate_with_margin(task, nb_classes)
    stats = client.last_margin_stats or {}
    rows.append({
        "client_id": client.id, "task": task,
        "prompt_gen": ptag, "ka_gen": ktag,
        "accuracy": f"{acc:.2f}",
        "margin_mean": f"{stats.get('mean', float('nan')):.4f}",
    })
    print(f"  client {client.id}: P={ptag} KA={ktag} -> acc={acc:.2f}")
    return acc


def main():
    args = parse_args()
    t = args.task
    old_path = (resolve(args.run_dir, args.old_ckpt) if args.old_ckpt
                else resolve(args.run_dir, f"checkpoints/task_{t:02d}_end.pth"))
    cur_path = resolve(args.run_dir, args.cur_ckpt)
    print(f"old generation: {old_path}\nnew generation: {cur_path}")

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=cur_path,
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes

    old_clients = {c["id"]: c for c in
                   load_checkpoint_file(old_path)["clients"]}

    rng_snapshot = rng_state_dict()
    q2_rows, q1_rows = [], []
    try:
        for client in server.clients:
            if client.task_id is None or client.task_id <= t:
                print(f"client {client.id}: task_id={client.task_id}, "
                      f"needs > {t}, skipped")
                continue
            old_state = old_clients.get(client.id)
            if (old_state is None
                    or old_state.get("tail_anchor_model") is None
                    or old_state.get("prompts") is None):
                print(f"client {client.id}: old generation missing, skipped")
                continue
            if client.heads[t] is None or t >= len(client.test_loader):
                print(f"client {client.id}: head/loader for task {t} "
                      "missing, skipped")
                continue

            # generations
            cur_ka_full = {k: v.clone() for k, v in
                           client.model.state_dict().items()}
            cur_ka = {"key": cur_ka_full["key"],
                      "anchor_pool": cur_ka_full["anchor_pool"]}
            old_ka_state = old_state["tail_anchor_model"]
            old_ka = {"key": old_ka_state["key"],
                      "anchor_pool": old_ka_state["anchor_pool"]}
            old_prompt = clone_prompt_template(client.vit)
            old_prompt.load_state_dict(old_state["prompts"])
            cur_prompt = client.prompts

            print(f"\n=== client {client.id} ===")
            key_old = old_ka["key"].to(client.device)
            key_new = cur_ka["key"].to(client.device)

            # ---------- Q2: 2x2 prompt x KA ----------
            for ptag, prompt in (("new", cur_prompt), ("old", old_prompt)):
                for ktag, ka in (("new", cur_ka), ("old", old_ka)):
                    eval_cell(client, t, nb_classes, ptag, prompt,
                              ktag, ka, q2_rows)

            # ---------- Q1: routing transfer ----------
            # deterministic pass over the private test set (no shuffle so
            # old/new see identical batches; vit train=True kept to mirror
            # the deployment forward, incl. batchwise prompt voting)
            test_data = client.test_loader[t]
            images = torch.stack([
                test_data[i][0] for i in range(len(test_data))])
            torch.manual_seed(9700 + 100 * client.id + t)
            z_old = extract_feats(client, old_prompt, images)
            torch.manual_seed(9700 + 100 * client.id + t)
            z_new = extract_feats(client, cur_prompt, images)

            r_old = routing_probs(client, z_old, key_old)
            r_new = routing_probs(client, z_new, key_new)

            # emergent-support rows: task-t TRAIN features top-1 routed
            # through K_old under the old generation (A2 protection-set
            # definition)
            train_idx = client.data_split_indices[t]["train"]
            base_ds = client.train_data[t]
            tr_images = torch.stack([base_ds[i][0] for i in train_idx])
            torch.manual_seed(9700 + 100 * client.id + t)
            z_tr_old = extract_feats(client, old_prompt, tr_images,
                                     batch_size=32)
            r_tr_old = routing_probs(client, z_tr_old, key_old)
            support_rows = set(r_tr_old.argmax(dim=1).tolist())
            in_support = torch.tensor(
                [row in support_rows for row in
                 r_old.argmax(dim=1).tolist()],
                device=r_old.device)

            for m in routing_metrics(r_old, r_new, in_support):
                q1_rows.append({
                    "client_id": client.id, "task": t,
                    "domain": "private_test", **m,
                    "n_support_rows": len(support_rows)})

            # stored A3 reference bank.
            # IMPORTANT: cur_ckpt is a round-boundary checkpoint saved
            # AFTER server prompt fusion. Therefore this measures
            # POST-FUSION deployed routing consistency and is NOT
            # expected to reproduce the local phase-2 compat_kl printed
            # during training (that one used the client's pre-fusion
            # local prompt).
            #
            # If post-fusion surrogate-bank KL is much larger than the
            # logged local phase-2 KL, server prompt fusion reintroduces
            # interface drift AFTER A3 (fusion breaks the compatibility
            # A3 just learned) -> fix the injection point, not the
            # mechanism.
            if (getattr(client, "_a3_teacher_probs", None) is not None
                    and getattr(client, "_a3_ref_raw", None) is not None):
                raw = client._a3_ref_raw
                idxs = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                bank_images = torch.stack([
                    client._a3_fetch_ref_images([i])[0]
                    for i in idxs])
                torch.manual_seed(9700 + 100 * client.id + t)
                z_bank_new = extract_feats(client, cur_prompt, bank_images,
                                           batch_size=16)
                r_bank_new = routing_probs(client, z_bank_new, key_new)
                r_bank_old = client._a3_teacher_probs.to(
                    r_bank_new.device)
                for m in routing_metrics(r_bank_old, r_bank_new):
                    q1_rows.append({
                        "client_id": client.id, "task": t,
                        "domain": "surrogate_bank", **m,
                        "n_support_rows": len(support_rows)})

            # restore deployment generation for the next client
            client.vit.load_prompts(cur_prompt)
            client.model.load_state_dict(cur_ka, strict=False)
    finally:
        restore_rng_state(rng_snapshot)

    # ---- write CSVs ----
    out_csv = args.out_csv or str(
        Path(args.run_dir) / "metrics" / "a3_counterfactual.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    for name, rows in (("a3_counterfactual_q2", q2_rows),
                       ("a3_counterfactual_q1", q1_rows)):
        path = str(Path(out_csv).with_name(name + ".csv"))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved {path} ({len(rows)} rows)")

    # ---- summary ----
    print("\n===== Q2: 2x2 prompt x KA (client-mean acc) =====")
    cells = {}
    for r in q2_rows:
        cells.setdefault((r["prompt_gen"], r["ka_gen"]),
                         []).append(float(r["accuracy"]))
    print(f"{'prompt':>6} {'KA':>4} {'acc':>7} {'n':>2}")
    for (ptag, ktag), vals in sorted(cells.items()):
        print(f"{ptag:>6} {ktag:>4} {np.mean(vals):>7.2f} {len(vals):>2}")

    print("\n===== Q1: routing KL / flip (client-mean) =====")
    print(f"{'domain':>14} {'split':>12} {'KL':>8} {'flip':>7} {'n':>5}")
    agg = {}
    for r in q1_rows:
        agg.setdefault((r["domain"], r["split"]), []).append(
            (r["kl_mean"], r["flip_rate"]))
    for (dom, split), vals in agg.items():
        kl = np.mean([v[0] for v in vals])
        fl = np.mean([v[1] for v in vals])
        print(f"{dom:>14} {split:>12} {kl:>8.3f} {fl:>7.3f} "
              f"{len(vals)*1:>5}")

    print("\nReading guide (Q1 is POST-FUSION, see script header):")
    print("  1. surrogate_bank KL >> logged local compat_kl (~4-5)"
          "  -> server prompt fusion breaks the compatibility A3 just"
          " learned; fix the injection point, not the mechanism")
    print("  2. surrogate_bank KL low, private_test KL high"
          "  -> surrogate reference mismatch (A3 protects the wrong"
          " distribution)")
    print("  3. both KL low but Q2 old/old >> new/new"
          "  -> routing consistency itself is insufficient for retention")
    print("  Q2 newP+oldKA drop -> prompt drift dominant; oldP+newKA drop "
          "-> Tail_Anchor-side (key/anchor) drift dominant (sub-split "
          "key vs anchor only if this cell dominates); both single drops "
          "small but new/new collapses -> cross-generation interaction")


if __name__ == "__main__":
    main()
