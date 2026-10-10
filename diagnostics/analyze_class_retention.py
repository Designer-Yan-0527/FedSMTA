"""Phase 1 diagnostic: 4-combo per-class channel decomposition.

    public/private scope  x  {old, current} prompt  x  {old, current} key/anchor

Adjudication target (FEDSMTA_ROADMAP.md §13, "federated-unique mechanism"):
2x2 attribution shows key/anchor interface mismatch dominates total
forgetting (58-96%), but KA drift is LOCAL component co-evolution that a
centralized CL system would also exhibit. The only federated-unique damage
channel is the AGGREGATED prompt (SIKF/BGPS mediates cross-client state).
Meanwhile per-class retention showed public classes (re-trained by other
clients) forget MORE than private ones (74.4% vs 83.9%). This script
decomposes that public-class excess damage per channel:

  - excess loss concentrated in the PROMPT combos (old prompt rescues
    public classes)  -> aggregation-mediated cross-client interface
    mismatch is a federated-unique mechanism (independent contribution);
  - excess loss equally present in the KA combos -> the mechanism is
    architecture-intrinsic; federation only modulates it (paper must
    downgrade the "federated-unique mechanism" claim).

For every (client, old task) it evaluates FOUR combinations, in the same
order as compatibility_drift.py, each with the same fixed shuffle seed
(9700+100*cid+old_task) so all combos see identical test batches:

    old/old      old prompt  + old key/anchor  (state at old task's end)
    old/cur      old prompt  + cur key/anchor   -> KA channel = oo - oc
    cur/old      cur prompt  + old key/anchor   ->  P channel = oo - co
    cur/cur      cur prompt  + cur key/anchor   (deployment; total = oo - cc)

Per-client accuracies under this script reproduce compatibility_drift.py
exactly (sample-weighted aggregation; local test sets are Dirichlet-cut to
class level, so unweighted class means are WRONG for cross-checks).

The public/private scope is self-evidenced by the run's
outer_split_manifest: a class appearing in >1 client is public (protocol
definition), in exactly 1 client is private. No external info needed.

Observer-only: consumes RNG (test shuffle) inside a fixed-seed guard,
snapshot/restore around the whole run. No training code is touched.

Output CSV: client_id, old_task, class_id, scope, n_clients, n_test,
prompt, key_anchor, acc  (one row per class per combo).

Usage (repo root, on the server):
  python diagnostics/analyze_class_retention.py \
      --run_dir output/cifar100/fedta/<run>/seed_42 \
      --old_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth \
      --new_ckpt checkpoints/task_04_end.pth \
      --out_csv diagnostics_output/class_retention_4combo.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diag_utils import bootstrap_server
from runtime_utils import (load_checkpoint_file, clone_prompt_template,
                           rng_state_dict, restore_rng_state)


def parse_args():
    p = argparse.ArgumentParser("Phase-1 per-class retention decomposition")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--old_ckpts", required=True,
                   help="comma-separated old checkpoints (END of each old "
                        "task), relative to run_dir or absolute")
    p.add_argument("--new_ckpt", default=None,
                   help="current checkpoint (default: checkpoints/latest.pth)")
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


def per_class_eval(client, task, nb_classes):
    """Replicates evaluate()/evaluate_with_margin() accuracy path, collecting
    per-sample (target, predict) pairs for per-class accuracy. Observer-only;
    consumes RNG (test shuffle) — the caller fixes the seed beforehand so the
    old/cur states see the identical batch order.

    Note (mask/target semantics, cf. Client_DF.evaluate): the head outputs
    nb_classes logits indexed by GLOBAL class id; masking to the task's
    classes keeps those indices, so both `predicts` and `target` are global
    class ids and can be grouped directly.
    """
    test_data = client.test_loader[task]
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

    client.model.load_head(client.heads[task])
    client.model.to(client.device)

    correct = {}
    total = {}
    for input, target in test_loader:
        input = input.to(client.device, non_blocking=True)
        target = target.to(client.device, non_blocking=True)
        with torch.no_grad():
            if client.original_model is not None:
                output = client.original_model(input)
                output = output['pre_logits'].requires_grad_(False)
                output = client.vit(input, task_id=client.task_id,
                                    cls_features=output, train=True)
                pre, _, _ = client.model(output['feat'].to(client.device),
                                          target.to(client.device))
        logits = pre
        mask = client.class_mask[task]
        not_mask = np.setdiff1d(np.arange(nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(client.device)
        logits = logits.index_fill(dim=1, index=not_mask,
                                   value=float('-inf'))
        predicts = torch.max(logits, dim=1)[1].cpu()
        for y, p in zip(target.cpu().tolist(), predicts.tolist()):
            total[y] = total.get(y, 0) + 1
            if y == p:
                correct[y] = correct.get(y, 0) + 1
    acc = {c: 100.0 * correct.get(c, 0) / total[c] for c in total}
    return acc, total


def norm_key(k):
    try:
        return str(int(k))
    except (TypeError, ValueError):
        return str(k)


def main():
    args = parse_args()
    old_paths = [resolve(args.run_dir, p) for p in args.old_ckpts.split(",")]
    new_path = (resolve(args.run_dir, args.new_ckpt)
                if args.new_ckpt
                else resolve(args.run_dir, "checkpoints/latest.pth"))

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=new_path,
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes

    new_ckpt = load_checkpoint_file(new_path)
    manifest = new_ckpt.get("outer_split_manifest")
    if not manifest:
        raise RuntimeError(
            "outer_split_manifest missing/empty in current checkpoint — "
            "public/private scope cannot be self-evidenced")

    # class -> set of owning clients (self-evidenced public/private scope)
    # manifest schema (cf. Server_DF.build_outer_split_manifest):
    #   {str(client_id): {str(task_id): {"raw_indices": [...],
    #                                    "class_mask": [...]}}}
    owners = {}
    for cid_key, tasks in manifest.items():
        for _tid, entry in tasks.items():
            for c in entry["class_mask"]:
                owners.setdefault(int(c), set()).add(norm_key(cid_key))

    rng_snapshot = rng_state_dict()
    rows = []
    try:
        for client in server.clients:
            cid = client.id
            # cur snapshot BEFORE any in-place load pollutes it (cf.
            # compatibility_drift alias note: state_dict shares storage)
            cur_ka = {k: v.clone() for k, v in client.model.state_dict().items()}
            cur_prompt = client.prompts  # module (or None), live current

            for old_path in old_paths:
                old_ckpt = load_checkpoint_file(old_path)
                old_state = old_ckpt["clients"][cid]
                old_task = old_state["task_id"]
                if old_task is None or old_task < 0:
                    continue
                if old_task >= len(client.test_loader) or \
                        client.heads[old_task] is None:
                    continue
                if old_state.get("prompts") is None:
                    continue
                print(f"client {cid} task {old_task}: evaluating "
                      f"({Path(old_path).name}) ...", flush=True)

                old_prompt = clone_prompt_template(client.vit)
                old_prompt.load_state_dict(old_state["prompts"])
                old_ka = old_state["tail_anchor_model"]

                # 4-combo evaluation, same order as compatibility_drift.py
                # (old/old, old/cur, cur/old, cur/cur). Fixed shuffle seed
                # per (client, old_task) => identical test batches across
                # all four combos.
                combos = []
                for p_name, prompt_mod in [("old", old_prompt),
                                           ("current", cur_prompt)]:
                    if prompt_mod is None:
                        continue  # no current prompts (unset live module)
                    for k_name, ka in [("old", old_ka), ("current", cur_ka)]:
                        client.vit.load_prompts(prompt_mod)
                        client.model.load_state_dict(
                            {"key": ka["key"],
                             "anchor_pool": ka["anchor_pool"]},
                            strict=False)
                        torch.manual_seed(9700 + 100 * cid + old_task)
                        acc_c, n_test = per_class_eval(
                            client, old_task, nb_classes)
                        combos.append((p_name, k_name, acc_c))

                # restore current components (cleanliness)
                if cur_prompt is not None:
                    client.vit.load_prompts(cur_prompt)
                client.model.load_state_dict(
                    {"key": cur_ka["key"], "anchor_pool": cur_ka["anchor_pool"]},
                    strict=False)

                for p_name, k_name, acc_c in combos:
                    for c in sorted(acc_c):
                        n_own = len(owners.get(c, set()))
                        scope = "public" if n_own > 1 else "private"
                        rows.append({
                            "client_id": cid,
                            "old_task": old_task,
                            "class_id": c,
                            "scope": scope,
                            "n_clients": n_own,
                            "n_test": n_test.get(c, 0),
                            "prompt": p_name,
                            "key_anchor": k_name,
                            "acc": f"{acc_c[c]:.2f}",
                        })
    finally:
        restore_rng_state(rng_snapshot)

    out_csv = args.out_csv or str(
        Path(args.run_dir) / "metrics" / "class_retention_4combo.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out_csv} ({len(rows)} rows)")

    # console summary: sample-weighted channel decomposition per
    # (old_task, scope) — the federated-unique-mechanism adjudication.
    #   P  channel = oo - co  (aggregated prompt drift; federated-unique)
    #   KA channel = oo - oc  (local key/anchor drift; architecture-intrinsic)
    #   total      = oo - cc
    def combo_stats(sel):
        """Sample-weighted accuracy per combo over selected rows."""
        out = {}
        for pn in ["old", "current"]:
            for kn in ["old", "current"]:
                vals = [(float(r["acc"]), int(r["n_test"]))
                        for r in sel
                        if r["prompt"] == pn and r["key_anchor"] == kn]
                n = sum(w for _, w in vals)
                out[(pn, kn)] = (sum(a * w for a, w in vals) / n
                                 if n else float("nan"))
        return out

    def print_decomp(tag, sel):
        if not sel:
            return None
        a = combo_stats(sel)
        oo, oc = a[("old", "old")], a[("old", "current")]
        co, cc = a[("current", "old")], a[("current", "current")]
        n_cls = len({(r["client_id"], r["class_id"]) for r in sel})
        n_test = sum(int(r["n_test"]) for r in sel
                     if r["prompt"] == "old" and r["key_anchor"] == "old")
        total = oo - cc
        p_share = (oo - co) / total if total > 0 else float("nan")
        ka_share = (oo - oc) / total if total > 0 else float("nan")
        print(f"{tag}: n_cls={n_cls} n_test={n_test}  "
              f"oo={oo:.2f} oc={oc:.2f} co={co:.2f} cc={cc:.2f}  "
              f"P={oo - co:+.2f}({p_share * 100:.0f}%) "
              f"KA={oo - oc:+.2f}({ka_share * 100:.0f}%) "
              f"total={total:+.2f}")
        return a

    print("\n===== 4-combo channel decomposition "
          "(sample-weighted acc %) =====")
    old_tasks = sorted({r["old_task"] for r in rows})
    for t in old_tasks:
        for scope in ["public", "private"]:
            sel = [r for r in rows
                   if r["old_task"] == t and r["scope"] == scope]
            print_decomp(f"task {t} {scope:7s}", sel)
    print("-" * 60)
    all_stats = {}
    for scope in ["public", "private"]:
        sel = [r for r in rows if r["scope"] == scope]
        all_stats[scope] = print_decomp(f"ALL {scope:7s}", sel)

    # adjudication verdict: where is the public-class excess damage?
    print("\n===== adjudication: public vs private channel shares =====")
    if all_stats.get("public") and all_stats.get("private"):
        verdicts = {}
        for scope in ["public", "private"]:
            a = all_stats[scope]
            total = a[("old", "old")] - a[("current", "current")]
            verdicts[scope] = {
                "P": a[("old", "old")] - a[("current", "old")],
                "KA": a[("old", "old")] - a[("old", "current")],
                "total": total,
            }
        p_pub, p_pri = verdicts["public"]["P"], verdicts["private"]["P"]
        ka_pub, ka_pri = verdicts["public"]["KA"], verdicts["private"]["KA"]
        print(f"public : total={verdicts['public']['total']:+.2f}  "
              f"P={p_pub:+.2f}  KA={ka_pub:+.2f}")
        print(f"private: total={verdicts['private']['total']:+.2f}  "
              f"P={p_pri:+.2f}  KA={ka_pri:+.2f}")
        p_excess = p_pub - p_pri
        ka_excess = ka_pub - ka_pri
        print(f"public-class excess damage: "
              f"P_excess={p_excess:+.2f}  KA_excess={ka_excess:+.2f}")
        if p_excess > ka_excess:
            print("=> excess damage rides the PROMPT (aggregated) channel: "
                  "federated-unique mechanism supported.")
        else:
            print("=> excess damage rides the KA (local) channel: mechanism "
                  "is architecture-intrinsic; federation only modulates it.")

    # task composition per client (public/private class counts per task)
    print("\n===== task composition (from manifest) =====")
    for cid_key, tasks in sorted(manifest.items(), key=lambda kv: norm_key(kv[0])):
        parts = []
        for tid, entry in sorted(tasks.items(), key=lambda kv: int(kv[0])):
            cls = [int(c) for c in entry["class_mask"]]
            n_pub = sum(1 for c in cls if len(owners.get(c, set())) > 1)
            parts.append(f"t{int(tid)}:{n_pub}pub/{len(cls) - n_pub}priv")
        print(f"client {norm_key(cid_key)}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
