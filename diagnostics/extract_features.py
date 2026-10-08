"""Phase 1 diagnostic: feature extraction.

Extracts, for every client and every seen task, two feature sets from the
SAME samples:
  A) z_ref : frozen pretrained ViT feature  (original_model output['feat'])
  B) z_op  : current prompt-enhanced feature BEFORE the Tail Anchor
             (client's vit output['feat'], i.e. the 768-d pre-anchor feature)

Per the Phase-1 spec, [z; anchor] mixed features are NEVER used here.

Output: one .npz (z_ref, z_op, labels, client_ids, task_ids, round_ids,
splits, sample_idx) plus a .json meta file (class masks, public classes,
run info, checkpoint round) consumable by analyze_multimodality.py /
synthetic_oracle_k2.py.

Usage (from repo root, on the server):
  python diagnostics/extract_features.py \
      --run_dir output/cifar100/fedta/<run_name>/seed_42 \
      --checkpoint checkpoints/task_04_end.pth \
      --out_dir diagnostics_output

Formal Phase-1 protocol defaults: --split train --batch_size 16 (only
override after the batch-sensitivity precheck justifies it).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server, resolve_checkpoint, REPO_ROOT


def parse_args():
    p = argparse.ArgumentParser("Phase-1 feature extraction")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{data_name}/{method}/{run_name}/seed_{seed})")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint path or name relative to run_dir "
                        "(default: checkpoints/latest.pth)")
    p.add_argument("--out_dir", default="diagnostics_output",
                   help="output directory (default: diagnostics_output)")
    p.add_argument("--split", default="train", choices=["train", "test", "both"],
                   help="which split of each client task to extract "
                        "(default: train -- the formal Phase-1 protocol; "
                        "test extraction is only for post-hoc reporting)")
    p.add_argument("--batch_size", default=16, type=int,
                   help="extraction batch size (default: 16 -- the formal "
                        "Phase-1 protocol; see the batch-sensitivity "
                        "precheck before changing)")
    p.add_argument("--data_path", default=None,
                   help="override dataset root (default: value from args.json)")
    p.add_argument("--device", default=None, help="override device (default: from args.json)")
    return p.parse_args()


def main():
    args_cli = parse_args()
    ckpt_path = resolve_checkpoint(args_cli.run_dir, args_cli.checkpoint)
    run_args, server = bootstrap_server(
        args_cli.run_dir, checkpoint=ckpt_path,
        data_path=args_cli.data_path, device=args_cli.device)
    device = torch.device(run_args.device)

    # round the checkpoint corresponds to (start_round = completed_round + 1
    # after load_checkpoint; -1 = initial state, no checkpoint loaded)
    completed_round = int(getattr(server, "start_round", 0)) - 1

    server.model.to(device)
    server.origin_model.to(device)
    server.origin_model.eval()

    splits = (["train", "test"] if args_cli.split == "both"
              else [args_cli.split])

    z_ref_all, z_op_all = [], []
    labels_all, clients_all, tasks_all = [], [], []
    splits_all, idx_all, rounds_all = [], [], []

    t0 = time.time()
    for client in server.clients:
        if client.task_id < 0:
            continue
        # the client's CURRENT prompts (end-of-training state)
        if client.prompts is not None:
            client.vit.load_prompts(client.prompts)
        for t in range(client.task_id + 1):
            if t not in client.data_split_indices:
                continue
            base_dataset = client.train_data[t]
            for split_name in splits:
                indices = list(client.data_split_indices[t][split_name])
                if not indices:
                    continue
                subset = Subset(base_dataset, indices)
                loader = DataLoader(subset, batch_size=args_cli.batch_size,
                                    shuffle=False,
                                    num_workers=run_args.num_workers)
                n_done = 0
                for inp, tgt in loader:
                    inp = inp.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)
                    with torch.no_grad():
                        out_ref = server.origin_model(inp)
                        z_ref = out_ref["feat"].float().cpu().numpy()
                        out_op = client.vit(
                            inp, task_id=t,
                            cls_features=out_ref["pre_logits"], train=True)
                        z_op = out_op["feat"].float().cpu().numpy()
                    b = z_ref.shape[0]
                    z_ref_all.append(z_ref)
                    z_op_all.append(z_op)
                    labels_all.append(tgt.cpu().numpy().astype(np.int64))
                    clients_all.append(np.full(b, client.id, dtype=np.int64))
                    tasks_all.append(np.full(b, t, dtype=np.int64))
                    splits_all.append(
                        np.full(b, 0 if split_name == "train" else 1,
                                dtype=np.int64))
                    rounds_all.append(
                        np.full(b, completed_round, dtype=np.int64))
                    idx_all.append(
                        np.asarray(indices[n_done:n_done + b], dtype=np.int64))
                    n_done += b
                print(f"client {client.id} task {t} {split_name}: "
                      f"{n_done} samples done ({time.time()-t0:.0f}s)")

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"features_{run_args.data_name}_{Path(ckpt_path).stem}"
    npz_path = out_dir / f"{stem}.npz"

    np.savez_compressed(
        npz_path,
        z_ref=np.concatenate(z_ref_all) if z_ref_all else np.zeros((0, 768), np.float32),
        z_op=np.concatenate(z_op_all) if z_op_all else np.zeros((0, 768), np.float32),
        labels=np.concatenate(labels_all) if labels_all else np.zeros((0,), np.int64),
        client_ids=np.concatenate(clients_all) if clients_all else np.zeros((0,), np.int64),
        task_ids=np.concatenate(tasks_all) if tasks_all else np.zeros((0,), np.int64),
        round_ids=np.concatenate(rounds_all) if rounds_all else np.zeros((0,), np.int64),
        splits=np.concatenate(splits_all) if splits_all else np.zeros((0,), np.int64),
        sample_idx=np.concatenate(idx_all) if idx_all else np.zeros((0,), np.int64),
    )

    # meta: class masks + public classes.
    # public/private is a property of the federated DATA PROTOCOL (class
    # masks), NOT of the extracted feature samples: a class is public iff
    # it appears in >= 2 clients' class masks. (The old fallback inferred
    # public classes from the extracted samples, which conflated the
    # protocol with feature-data statistics.)
    class_to_clients = {}
    for c in server.clients:
        this_client_classes = set()
        for task_classes in c.class_mask:
            this_client_classes.update(int(x) for x in task_classes)
        for cls in this_client_classes:
            class_to_clients.setdefault(cls, set()).add(c.id)
    public_classes = sorted(
        cls for cls, cs in class_to_clients.items() if len(cs) >= 2)

    meta = {
        "run_dir": str(Path(args_cli.run_dir).resolve()),
        "checkpoint": str(ckpt_path),
        "round": completed_round,
        "data_name": run_args.data_name,
        "nb_classes": run_args.nb_classes,
        "seed": run_args.seed,
        "client_num": run_args.client_num,
        "task_num": run_args.task_num,
        "client_class_masks": {
            str(c.id): [list(map(int, t)) for t in c.class_mask]
            for c in server.clients},
        "public_classes": public_classes,
        "split_encoding": {"0": "train", "1": "test"},
        "feature_note": "z_ref = frozen pretrained ViT feat; "
                        "z_op = current prompt-enhanced pre-anchor feat",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved {npz_path}")
    print(f"public classes ({len(public_classes)}): {public_classes}")


if __name__ == "__main__":
    main()
