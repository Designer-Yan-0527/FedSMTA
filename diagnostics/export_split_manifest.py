"""Export the frozen CIFAR-100 federated split manifest (baseline protocol v1).

Single source of truth that external baselines (FedMGP -> C2Prompt -> GLFC)
must consume, so that every method sees the SAME client/task/class/sample
assignment instead of its own same-seed random_split (different codebases
consume RNG in different orders, so seed=42 alone guarantees nothing).

Three layers, matching how FedSMTA/FedTA actually consume data:

  outer   per (client, task): raw train-set indices + class mask.
          Produced by cifar100_Data_Spliter.random_split(), which touches
          ONLY the python `random` and numpy streams. Regenerated here
          standalone with the exact main.build_server RNG order:
          seed -> random_split() -> process_testdata().
          (The create_model call in between touches only the torch RNG.)
  surrogate / test_pool
          process_testdata outputs (indices into the CIFAR-100 test set;
          20 + 80 per class). NOTE: iCIFAR100c(subset=...) unwraps
          CustomedSubset.dataset, so the surrogate pool actually used at
          runtime is the FULL test set in both codebases (upstream FedTA
          quirk, preserved verbatim); the indices are recorded anyway for
          completeness and bit-level bookkeeping.
  inner   per (client, task): 70/30 positions INTO the outer CustomedSubset
          (torch.utils.data.random_split semantics). The inner 30% subsets
          ARE the evaluation sets of both codebases. They depend on the
          torch RNG stream interleaved with training, so they are taken
          from a reference run checkpoint's data_split_state, not
          regenerated.

Seals (fail-loud, no vacuous pass):
  S1  regenerated outer == reference checkpoint outer_split_manifest
      (bit-identical; proves the standalone regeneration IS the run's split)
  S2  inner entries: exact 70/30 sizes (int(n*0.7) convention), disjoint,
      covering all positions of the corresponding outer subset
  S3  surrogate + test_pool partition the full 10k test set exactly
  S4  (optional, --verify-task0-replay) replay build_server + round-0
      update_data on CPU with the fedta method and require the resulting
      task-0 inner splits == manifest inner task 0 (proves the reference
      checkpoint's torch RNG stream through the first task boundary equals
      the fedta baseline's).

Reference-checkpoint choice: any seed-42 run whose torch RNG stream through
every task boundary equals the fedta baseline's. The A2 arms qualify: all
three arms' data_split_state are bit-identical (empirically verified), and
A2's only extra work vs fedta is the post-step parameter restore plus the
RNG-snapshot-guarded feature pass (RNG-neutral by construction, cf.
Models/Client_DF.py). When the true fedta seed_42 checkpoint becomes
available, re-run with it via --reference-checkpoint to upgrade the
provenance; S1/S2 must still pass.

Usage (FedSMTA conda env, from the repo root):

  python diagnostics/export_split_manifest.py \
      --reference-checkpoint output/cifar100/fedta_a2/a2_anchor/seed_42/checkpoints/latest.pth \
      -o protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      --verify-task0-replay
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def torch_load_compat(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location="cpu")


def normalize_outer(manifest):
    """int client/task keys -> str, values as plain int lists."""
    out = {}
    for cid, tasks in manifest.items():
        out[str(cid)] = {}
        for tid, entry in tasks.items():
            out[str(cid)][str(tid)] = {
                "raw_indices": [int(x) for x in entry["raw_indices"]],
                "class_mask": [int(c) for c in entry["class_mask"]],
            }
    return out


def normalize_inner(dss):
    """checkpoint data_split_state (int or str keys) -> normalized dict."""
    out = {}
    for cid, tasks in (dss or {}).items():
        out[str(cid)] = {}
        for tid, entry in tasks.items():
            out[str(cid)][str(tid)] = {
                "train": [int(x) for x in entry["train"]],
                "test": [int(x) for x in entry["test"]],
            }
    return out


def fail(msg):
    raise SystemExit(f"[export_split_manifest] FAIL: {msg}")


def verify_task0_replay(args, manifest):
    """Seal S4: replay the fedta stream through the round-0 update_data on
    CPU and require the task-0 inner splits to equal the manifest's."""
    import argparse as _argparse
    from config.datasets_delay import get_args_parser
    from main import build_server

    parser = _argparse.ArgumentParser("replay")
    sub = parser.add_subparsers(dest="subparser_name")
    cp = sub.add_parser("datasets_delay")
    get_args_parser(cp)
    replay_args = parser.parse_args([
        "datasets_delay", "--method", "fedta", "--device", "cpu",
        "--data_path", args.data_path, "--seed", str(args.seed),
        "--client_num", str(args.client_num), "--task_num", str(args.task_num),
        "--private_class_num", str(args.private_class_num),
        "--surrogate_num", str(args.surrogate_num),
    ])
    server = build_server(replay_args, run_manager=None)
    server.init_client()
    # round 0: all clients' update_data (-> get_data(0)) run before any
    # training, exactly like Server_DF.train_clients
    for j in range(server.client_num):
        server.clients[j].update_data(round=0, args=replay_args)

    for j in range(server.client_num):
        got = server.clients[j].data_split_indices.get(0)
        if got is None:
            fail(f"task0 replay: client {j} produced no task-0 split")
        want = manifest["inner"][str(j)]["0"]
        if [int(x) for x in got["train"]] != want["train"] or \
           [int(x) for x in got["test"]] != want["test"]:
            fail(f"task0 replay: client {j} inner split != manifest "
                 f"(the reference checkpoint's torch RNG stream diverges "
                 f"from the fedta baseline's)")
    print("[seal S4] task-0 inner split replay (fedta stream): PASS")


def main():
    p = argparse.ArgumentParser("export the frozen split manifest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--client_num", type=int, default=5)
    p.add_argument("--task_num", type=int, default=5)
    p.add_argument("--private_class_num", type=int, default=15)
    p.add_argument("--surrogate_num", type=int, default=20)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--reference-checkpoint", required=True,
                   help="seed-matching run checkpoint providing the inner "
                        "70/30 splits (data_split_state) and the outer "
                        "cross-check (outer_split_manifest)")
    p.add_argument("--verify-task0-replay", action="store_true",
                   help="slow seal S4: CPU replay of the fedta stream "
                        "through round-0 update_data")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    import numpy as np

    # ---- regenerate outer + surrogate/test pool (exact RNG order) ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    from data.cifar100_subset_spliter import cifar100_Data_Spliter

    spliter = cifar100_Data_Spliter(
        client_num=args.client_num, task_num=args.task_num,
        private_class_num=args.private_class_num,
        input_size=args.input_size, data_path=args.data_path)
    client_data, client_mask = spliter.random_split()
    spliter2 = cifar100_Data_Spliter(
        client_num=args.client_num, task_num=args.task_num,
        private_class_num=args.private_class_num,
        input_size=args.input_size, data_path=args.data_path)
    surro_data, test_pool = spliter2.process_testdata(args.surrogate_num)

    outer = {}
    for i in range(args.client_num):
        outer[str(i)] = {}
        for t in range(args.task_num):
            outer[str(i)][str(t)] = {
                "raw_indices": [int(x) for x in client_data[i][t].indices],
                "class_mask": [int(c) for c in client_mask[i][t]],
            }
    surrogate_indices = [int(x) for x in surro_data.indices]
    test_pool_indices = [int(x) for x in test_pool.indices]

    # ---- reference checkpoint ----
    ckpt = torch_load_compat(args.reference_checkpoint)
    ref_args = ckpt.get("args", {})
    for key, want in (("seed", args.seed), ("client_num", args.client_num),
                      ("task_num", args.task_num),
                      ("private_class_num", args.private_class_num),
                      ("surrogate_num", args.surrogate_num)):
        got = ref_args.get(key)
        if got is not None and int(got) != int(want):
            fail(f"reference checkpoint {key}={got} != export {key}={want}")

    # ---- seal S1: outer == checkpoint outer_split_manifest ----
    if "outer_split_manifest" not in ckpt:
        fail("reference checkpoint missing outer_split_manifest")
    ref_outer = normalize_outer(ckpt["outer_split_manifest"])
    if outer != ref_outer:
        # locate the first divergence for a usable error message
        for cid in outer:
            for tid in outer[cid]:
                if outer[cid][tid] != ref_outer.get(cid, {}).get(tid):
                    fail(f"outer split diverges at client {cid} task {tid}: "
                         f"regenerated partition != reference run partition")
        fail("outer split differs from reference checkpoint (size mismatch)")
    print("[seal S1] regenerated outer == reference outer_split_manifest: "
          "PASS")

    # ---- inner from checkpoint ----
    inner = normalize_inner(ckpt.get("data_split_state"))
    if not inner:
        fail("reference checkpoint has empty data_split_state")

    # ---- seal S2: inner structural validation ----
    for cid in outer:
        for tid in outer[cid]:
            entry = inner.get(cid, {}).get(tid)
            if entry is None:
                fail(f"inner split missing for client {cid} task {tid} "
                     f"(reference run did not reach this task?)")
            n = len(outer[cid][tid]["raw_indices"])
            n70 = int(n * 0.7)
            tr, te = entry["train"], entry["test"]
            if len(tr) != n70 or len(te) != n - n70:
                fail(f"inner split client {cid} task {tid}: sizes "
                     f"{len(tr)}/{len(te)} != expected {n70}/{n - n70}")
            if sorted(tr + te) != list(range(n)):
                fail(f"inner split client {cid} task {tid}: not a disjoint "
                     f"cover of the outer subset positions")
    print("[seal S2] inner 70/30 sizes + disjoint cover: PASS")

    # ---- seal S3: surrogate + test pool partition the test set ----
    if sorted(surrogate_indices + test_pool_indices) != list(range(10000)):
        fail("surrogate + test_pool do not partition the 10k test set")
    n_per_class = len(surrogate_indices) // 100
    print(f"[seal S3] surrogate({len(surrogate_indices)}, "
          f"{n_per_class}/class) + test_pool({len(test_pool_indices)}) "
          f"partition the test set: PASS")

    manifest = {
        "schema": "fedsmta-split-manifest-v1",
        "dataset": "cifar100",
        "seed": args.seed,
        "protocol": {
            "client_num": args.client_num,
            "task_num": args.task_num,
            "private_class_num": args.private_class_num,
            "surrogate_num": args.surrogate_num,
            "input_size": args.input_size,
            "nb_classes": 100,
        },
        "outer": outer,
        "surrogate": {"indices": surrogate_indices},
        "test_pool": {"indices": test_pool_indices},
        "inner": inner,
        "provenance": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": "diagnostics/export_split_manifest.py",
            "outer_source": "standalone regeneration (python random + numpy "
                            "streams, exact main.build_server RNG order)",
            "inner_source": str(Path(args.reference_checkpoint).resolve()),
            "seals": ["S1 outer==checkpoint manifest", "S2 inner structure",
                      "S3 test-set partition"],
        },
    }

    if args.verify_task0_replay:
        verify_task0_replay(args, manifest)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    n_train = sum(len(outer[c][t]["raw_indices"])
                  for c in outer for t in outer[c])
    print(f"\n[export_split_manifest] wrote {out}")
    print(f"  outer train samples: {n_train} "
          f"(official dirichlet int() truncation drops "
          f"{50000 - n_train} samples; identical in both codebases)")
    print(f"  clients/tasks: {args.client_num} x {args.task_num}")
    print(f"  inner source: {manifest['provenance']['inner_source']}")


if __name__ == "__main__":
    main()
