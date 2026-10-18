"""Gate for the C2Prompt protocol adapter (run_c2p.py).

Part A (informational, no pass/fail): what an UNADAPTED same-seed C2Prompt
run would generate under the frozen run params (epochs_global=25,
tasks_global=5, num_clients=5, sim=0) vs the frozen manifest. The official
ImageNet-R protocol is structurally different (500 virtual classes over
200 real, 20-class tasks, 20% data slices, sim-resampled real
distribution); Part A quantifies the gap that motivates the adapter.

Part B (gating, must all pass):
  B1 manifest structure: 25 client-tasks, 8-class masks, per-client task
     masks disjoint, inner 70/30 sizes + disjoint cover
  B2 class_distribution injection: the manifest-derived dicts exactly
     equal the manifest masks (identity real mapping, [0,1] proportion)
  B3 dataset patches: patched iIMAGENET_R builds 24000 train / 6000 test
     paths in the FedTA ordering (class counts match the folder layout);
     getTrainData(client, task mask) returns exactly the inner-70% train
     samples of the frozen outer subset (labels = raw-index targets);
     get_image_class_dict agrees with getTrainData per class;
     getTestData_hard returns exactly the test_pool samples per class
  B4 source-patch seal: every surgical replacement in patch_main_source
     applies exactly once (twice for the real-dict append) and the
     patched main.py still compiles
  B5 frozen-argv seal: build_argv's frozen protocol values equal the
     manifest protocol (clients/tasks/classes/seed + the 25-round
     cadence), so a hand-edited drift in the adapter's hardcoded argv
     cannot silently run off-protocol

Usage (FedSMTA conda env, repo root):

  python protocol_adapters/gate_c2p.py \
      --manifest protocol_manifests/inr_c5t5p40s5_seed42.json \
      --data-root local_datasets/imagenet-r \
      -o protocol_adapters/gate_c2p_report.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import run_c2p  # noqa: E402

C2P_ROOT = REPO_ROOT / "external_baselines" / "NeurIPS2025-C2Prompt"


def fail(msg):
    raise SystemExit(f"[gate_c2p] FAIL: {msg}")


def part_a(m):
    """Native (unadapted) C2Prompt ImageNet-R distribution under the frozen
    run params - informational."""
    seed = m["seed"]
    random.seed(seed)
    tasks_per_client = 5
    num_clients = 5
    native_client, native_real, native_prop = {}, {}, {}
    for i in range(num_clients):
        task_list = []
        for j in range(tasks_per_client):
            task_list.append(list(range((i * tasks_per_client + j) * 20,
                                        (i * tasks_per_client + j + 1) * 20)))
        native_client[i] = task_list
    # faithful re-implementation of main.py L51-70 (sim=0 official default:
    # similar_level ~ U{0..20}, so the sim-mixing branch fires 20/21 of the
    # time — that IS native behavior and exactly why native != manifest)
    for i in range(num_clients):
        task_list = []
        for j in range(tasks_per_client):
            if i * tasks_per_client + j > 0:
                similar_global_task_id = random.sample(
                    list(range(i * tasks_per_client + j)), 1)[0]
                similar_level = random.sample(list(range(0, 21)), 1)[0]
                if similar_level != 0:
                    similar_client_id = int(similar_global_task_id
                                            // tasks_per_client)
                    similar_task_id = int(similar_global_task_id
                                          % tasks_per_client)
                    if similar_client_id == i:
                        base = task_list[similar_task_id]
                    else:
                        base = native_real[similar_client_id][similar_task_id]
                    task_list.append(
                        list(random.sample(base, similar_level))
                        + list(random.sample(
                            list(set(list(range(200))) - set(base)),
                            20 - similar_level)))
                else:
                    task_list.append(list(random.sample(list(range(200)), 20)))
            else:
                task_list.append(list(random.sample(list(range(200)), 20)))
        native_real[i] = task_list
    for i in range(num_clients):
        task_list = []
        for j in range(tasks_per_client):
            proportion_list = [0, 0.2, 0.4, 0.6, 0.8]
            start = proportion_list[(i * tasks_per_client + j) % 5]
            task_list.append([start, start + 0.2])
        native_prop[i] = task_list

    overlaps = []
    for i in range(num_clients):
        for j in range(tasks_per_client):
            mask = set(m["outer"][str(i)][str(j)]["class_mask"])
            overlaps.append(len(mask & set(native_real[i][j])))
    info = {
        "verdict": "informational (expected: native != manifest)",
        "native_structure": {
            "classes_per_task": 20, "virtual_class_space": 500,
            "real_class_space": 200, "tasks_per_client": tasks_per_client,
            "data_slice_per_task": "20% of each class (proportion)",
            "sim": 0,
        },
        "manifest_structure": {
            "classes_per_task": 8, "class_space": 200,
            "tasks_per_client": tasks_per_client,
            "data_per_task": "inner-70% of the frozen outer subset",
        },
        "mean_overlap_native_real_vs_manifest_mask":
            sum(overlaps) / len(overlaps),
        "note": "a same-seed unadapted run sees a different partition; "
                "the adapter is required for sample-assignment equality",
    }
    print(f"[part A] native vs manifest mean class overlap "
          f"(of 8-class masks vs native 20-class real sets): "
          f"{info['mean_overlap_native_real_vs_manifest_mask']:.2f} "
          f"[informational]")
    return info


def part_b(m, data_root):
    """Gating checks on manifest, injection, dataset patches, source
    patches. Returns report dict; raises SystemExit on any failure."""
    import numpy as np

    # ---- B1 manifest structure ----
    for cid in range(5):
        masks = []
        for tid in range(5):
            entry = m["outer"][str(cid)][str(tid)]
            if len(entry["class_mask"]) != 8:
                fail(f"B1: client {cid} task {tid} mask has "
                     f"{len(entry['class_mask'])} classes, expected 8")
            masks.append(set(entry["class_mask"]))
            n = len(entry["raw_indices"])
            inner = m["inner"][str(cid)][str(tid)]
            n70 = int(n * 0.7)
            if len(inner["train"]) != n70 or \
               len(inner["test"]) != n - n70:
                fail(f"B1: inner sizes wrong for client {cid} task {tid}")
            if sorted(inner["train"] + inner["test"]) != list(range(n)):
                fail(f"B1: inner not a disjoint cover for client {cid} "
                     f"task {tid}")
        for a in range(5):
            for b in range(a + 1, 5):
                if masks[a] & masks[b]:
                    fail(f"B1: client {cid} tasks {a}/{b} masks overlap")
    print("[part B1] manifest structure (25x8 masks, disjoint per client, "
          "inner 70/30 cover): PASS")

    # ---- B2 class_distribution injection (exec the PATCHED source's
    # ImageNet_R branch standalone and compare against the manifest) ----
    root = str(C2P_ROOT.resolve())
    src_path = root + "/main.py"
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    patched = run_c2p.patch_main_source(src)  # raises if counts wrong
    start = patched.index("    if args.dataset == 'ImageNet_R':")
    end = patched.index("    elif args.dataset == 'DomainNet':")
    snippet = patched[start:end]
    import textwrap
    class _Args:
        pass
    _a = _Args()
    _a.dataset = "ImageNet_R"
    _a.num_clients = 5
    _a.epochs_global = 25
    _a.tasks_global = 5
    _a.sim = 0
    _ns = {"args": _a, "_C2P_MANIFEST": m, "random": random}
    exec(textwrap.dedent(snippet), _ns)
    for i in range(5):
        for j in range(5):
            want = list(m["outer"][str(i)][str(j)]["class_mask"])
            if _ns["class_distribution_client"][i][j] != want:
                fail(f"B2: class_distribution_client mismatch at {i}/{j}")
            if _ns["class_distribution_client_real"][i][j] != want:
                fail(f"B2: class_distribution_client_real mismatch at "
                     f"{i}/{j}")
            if _ns["class_distribution_client_proportion"][i][j] != [0.0, 1.0]:
                fail(f"B2: proportion mismatch at {i}/{j}")
    print("[part B2] patched-source class_distribution (client/real/"
          "proportion) == manifest masks / [0,1]: PASS")

    # ---- B3 dataset patches ----
    root = str(C2P_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import os
    os.chdir(root)
    run_c2p.install_nni_stub()
    run_c2p.build_state(data_root)
    iIMAGENET_R = run_c2p.patch_dataset()

    from dataloaders.utils import get_transform
    train_tf = get_transform(dataset="ImageNet_R", phase="train", aug=True,
                             resize_imnet=True)
    test_tf = get_transform(dataset="ImageNet_R", phase="test", aug=True,
                            resize_imnet=True)

    train_ds = iIMAGENET_R(str(Path(data_root).resolve()), train=True,
                           transform=train_tf, download_flag=False,
                           seed=m["seed"], validation=False)
    test_ds = iIMAGENET_R(str(Path(data_root).resolve()), train=False,
                          transform=test_tf, download_flag=False,
                          seed=m["seed"], validation=False)
    if len(train_ds.data) != 24000 or len(test_ds.data) != 6000:
        fail(f"B3: expected 24000/6000 paths, got "
             f"{len(train_ds.data)}/{len(test_ds.data)}")
    counts = np.bincount(np.asarray(train_ds.targets))
    if len(counts) != 200 or (counts == 0).any():
        fail("B3: train class counts wrong")
    print(f"[part B3a] patched iIMAGENET_R: 24000 train / 6000 test paths, "
          f"200 classes (min class count {counts.min()}, max "
          f"{counts.max()}): PASS")

    # getTrainData == inner-70 train samples for every (client, task)
    for cid in range(5):
        for tid in range(5):
            mask = list(m["outer"][str(cid)][str(tid)]["class_mask"])
            train_ds.getTrainData(mask, [], [], cid, classes_real=mask,
                                  classes_proportion=[0.0, 1.0],
                                  class_distribution_client_di=None)
            raws = [int(m["outer"][str(cid)][str(tid)]["raw_indices"][pos])
                    for pos in m["inner"][str(cid)][str(tid)]["train"]]
            want_labels = sorted(int(train_ds.targets[r]) for r in raws)
            got_labels = sorted(int(x) for x in train_ds.TrainLabels)
            if len(train_ds.TrainData) != len(raws) or \
               got_labels != want_labels:
                fail(f"B3: getTrainData client {cid} task {tid} != inner-70"
                     f" train samples ({len(train_ds.TrainData)} vs "
                     f"{len(raws)})")
    print("[part B3b] getTrainData == inner-70% train samples of the "
          "frozen outer subsets, all 25 client-tasks: PASS")

    # get_image_class_dict agrees per class
    for cid in range(5):
        per = run_c2p._STATE["client_class_train_idx"][cid]
        for label, raws in per.items():
            got = train_ds.get_image_class_dict(label, cid, [0.0, 1.0])
            want = [str(train_ds.data[r]) for r in raws]
            if sorted(got) != sorted(want):
                fail(f"B3: get_image_class_dict client {cid} class {label} "
                     "mismatch")
    print("[part B3c] get_image_class_dict == per-(client,class) manifest "
          "train samples: PASS")

    # getTestData_hard == test_pool samples per class
    for label in range(0, 200, 40):
        test_ds.getTestData_hard([label], [label])
        want = run_c2p._STATE["test_pool_by_class"].get(label, [])
        if len(test_ds.TestData) != len(want):
            fail(f"B3: getTestData_hard class {label}: "
                 f"{len(test_ds.TestData)} != {len(want)} test-pool samples")
        got_paths = set(str(p) for p in test_ds.TestData)
        want_paths = set(str(test_ds.data[r]) for r in want)
        if got_paths != want_paths:
            fail(f"B3: getTestData_hard class {label}: sample mismatch")
    n_pool = len(m["test_pool"]["indices"])
    print(f"[part B3d] getTestData_hard == manifest test_pool samples "
          f"(spot check every 40th class; pool = {n_pool} of 6000): PASS")

    # ---- B4 source-patch seal ----
    src_path = root + "/main.py"
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    patched = run_c2p.patch_main_source(src)  # raises if counts wrong
    compile(patched, src_path, "exec")        # raises if it no longer parses
    n_expected = sum(c for _, _, c in run_c2p.SOURCE_PATCHES)
    print(f"[part B4] main.py source patches: {len(run_c2p.SOURCE_PATCHES)} "
          f"replacements ({n_expected} sites), patched source compiles: "
          f"PASS")
    hard_path = root + "/models_Cprompt/vit_coda_p_hard.py"
    with open(hard_path, "r", encoding="utf-8") as f:
        hard_src = f.read()
    hard_patched = run_c2p.patch_hard_source(hard_src)
    compile(hard_patched, hard_path, "exec")
    n_hard = sum(c for _, _, c in run_c2p.HARD_SOURCE_PATCHES)
    print(f"[part B4] vit_coda_p_hard.py source patches: "
          f"{len(run_c2p.HARD_SOURCE_PATCHES)} replacements ({n_hard} "
          f"sites), patched source compiles: PASS")

    # ---- B5 frozen-argv seal ----
    argv = run_c2p.build_argv(data_root, [], m)
    flags, i = {}, 0
    while i < len(argv):
        if argv[i] == "--prompt_param":  # nargs: 9 values
            flags["prompt_param"] = argv[i + 1:i + 10]
            i += 10
            continue
        flags[argv[i][2:]] = argv[i + 1]
        i += 2
    p = m["protocol"]
    want = {
        "dataset": "ImageNet_R",
        "num_clients": str(p["client_num"]),
        "local_clients": str(p["client_num"]),
        "tasks_global": str(p["task_num"]),
        "epochs_global": str(5 * p["task_num"]),  # 5 rounds/task cadence
        "numclass": str(p["nb_classes"]),
        "num_classes": str(p["nb_classes"]),
        "class_per_task": "8",   # == mask size, enforced by B1
        "task_size": "8",
        "seed": str(m["seed"]),
    }
    for k, v in want.items():
        if flags.get(k) != v:
            fail(f"B5: frozen argv --{k} is {flags.get(k)!r}, expected "
                 f"{v!r} (manifest protocol drift)")
    if flags.get("prompt_param") != ["25", "10", "10", "8", "0", "0", "6",
                                     "10", "8"]:
        fail(f"B5: prompt_param is {flags.get('prompt_param')!r}, expected "
             "the official scripts/run_imagenetr.sh values")
    print("[part B5] build_argv frozen values == manifest protocol "
          "(clients/tasks/classes/seed/cadence + official prompt_param): "
          "PASS")

    return {
        "B1_manifest_structure": "PASS",
        "B2_class_distribution_injection": "PASS",
        "B3_dataset_patches": "PASS",
        "B4_source_patch_seal": "PASS",
        "B5_frozen_argv_seal": "PASS",
    }


def main():
    p = argparse.ArgumentParser("gate for the C2Prompt protocol adapter")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root",
                   default=str(REPO_ROOT / "local_datasets" / "imagenet-r"))
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()
    args.data_root = str(Path(args.data_root).resolve())  # part_b chdirs
                                                     # into the C2Prompt root
    if args.output:
        args.output = str(Path(args.output).resolve())  # ditto

    m = run_c2p.load_manifest(args.manifest)

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "part_a_native_vs_manifest": part_a(m),
        "part_b": part_b(m, args.data_root),
    }
    print("\n[gate_c2p] ALL GATING CHECKS PASSED")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[gate_c2p] wrote {out}")


if __name__ == "__main__":
    main()
