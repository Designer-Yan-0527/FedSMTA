"""C2Prompt (NeurIPS2025, external_baselines/NeurIPS2025-C2Prompt) training
entry with the frozen protocol manifest injected.

Frozen design (external-baseline fairness policy, same as FedMGP):

  The official C2Prompt implementation is preserved verbatim; ONLY the
  dataset-partition interface is replaced so C2Prompt consumes the exact
  same client/task/class/sample assignment as FedSMTA. No file inside
  external_baselines/NeurIPS2025-C2Prompt is ever edited; all patches are
  applied at runtime from this module.

Why ImageNet-R (not CIFAR-100): the official repo's CIFAR-100 path is dead
code (main.py's class_distribution only generates ImageNet_R/DomainNet;
prompt_hard.py's getTrainData call signature only matches iDOMAINNET's
iDataset, not iCIFAR100). The official ImageNet-R protocol
(scripts/run_imagenetr.sh) is the supported path, so C2Prompt is aligned
on ImageNet-R and the CIFAR-100 main table simply has no C2Prompt row.

Injection points (all fail-loud):

  1. main.py class_distribution client/real/proportion dicts
       -> manifest outer class_masks (identity real mapping, [0,1]
          proportion passthrough). Source-level surgical replacement of
          the three dict-building loops inside the ImageNet_R branch.
  2. iIMAGENET_R.load()
       -> builds self.data/self.targets as file paths from OUR
          local_datasets/imagenet-r/{train,test} layout (sorted folders,
          sorted files = the exact ordering data.continual_datasets
          .Imagenet_R uses, i.e. the manifest raw_indices domain). The
          official yaml splits are unusable here (absolute foreign paths,
          different 80/20 assignment).
  3. iDataset.getTrainData / get_image_class_dict
       -> per-(client, class) sample lists derived from the manifest
          (inner-70% TRAIN positions of the frozen outer subsets: the
          exact samples FedTA/FedMGP clients train on). Task-level
          queries fall out for free because per-client task masks are
          disjoint.
  4. iDataset.getTestData_hard
       -> manifest test_pool (test set minus the 5/class surrogate
          samples FedTA consumes as surrogate-phase training data).
          C2Prompt's native eval (model_global_eval_hard over all classes
          trained so far) is preserved; only the sample pool is frozen.
  5. torch.random.fork_rng(devices=[3])  -> devices=[] (single-GPU; the
          official line forks GPU 3 which does not exist on 1-GPU hosts)
  6. Task progression: the official 50%-of-clients-advance rule (plus the
          hardcoded ImageNet_R 2-client switches keyed on ep_g//3) is
          replaced by full progression (all 5 clients advance through all
          5 tasks), matching the FedTA synchronous protocol the manifest
          froze. Without this, 15 of the 25 frozen client-tasks would
          never train.
  7. nni stub (get_next_parameter -> {}, report_* -> no-op), same pattern
     as the FedMGP deepluke stub.
  8. metrics recorder (protocol_adapters/metrics_recorder.py): OBSERVES
     the native task-boundary eval (which already evaluates every trained
     client-task on the manifest test pool) and records
     (client_id, after_task, test_task, accuracy) into the RunManager-
     format CSV tree output/ImageNet-R/c2prompt/manifest_adapter/
     seed_{seed}/metrics/. Pure observation: zero extra computation, zero
     RNG impact, official stdout/log_train.txt unchanged.

Protocol mapping (frozen, enforced against the manifest):
  dataset=ImageNet_R, num_clients=5, tasks_global=5 (5 tasks/client),
  epochs_global=25 (5 rounds/task = the official 15/3 cadence),
  local_clients=5 (full participation), numclass=num_classes=200,
  class_per_task=task_size=8, seed=manifest seed.
Method-native (official scripts/run_imagenetr.sh, overridable):
  method=cprompt_c2prompt_v2, prompt_flag=codap_2d_v2,
  prompt_param=[25,10,10,8,0,0,6,10,8] (e_task_number=25 = exactly our
  5x5=25 global tasks), learning_rate=0.005, batch_size=64,
  epochs_local=10, global_update_lr=100.

Budget note: the exemplar/replay code paths in C2Prompt are dead for the
v2 method (update_new_set is extension-only; _get_train_and_test_dataloader
is always called with mix=False), so C2Prompt trains on current-task data
only -> Replay/Memory column: none (method-native behavior).

Usage (from the FedSMTA repo root, FedSMTA conda env; needs pyyaml,
transformers, torch_optimizer installed - all unused-import-level deps):

  python protocol_adapters/run_c2p.py \
      --manifest protocol_manifests/inr_c5t5p40s5_seed42.json \
      [--data-root local_datasets/imagenet-r] [--device 0] \
      [forwarded C2Prompt flags, e.g. --epochs_local 10]
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import metrics_recorder  # noqa: E402

C2P_ROOT = REPO_ROOT / "external_baselines" / "NeurIPS2025-C2Prompt"

# ---------------------------------------------------------------------------
# adapter state
# ---------------------------------------------------------------------------

_STATE = {
    "manifest": None,
    # client -> class -> sorted raw train-set indices (inner-70% positions)
    "client_class_train_idx": None,
    # class -> sorted raw test-pool indices
    "test_pool_by_class": None,
}


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema") != "fedsmta-split-manifest-v1":
        raise ValueError(f"{path}: unsupported schema {m.get('schema')!r}")
    if m.get("dataset") != "ImageNet-R":
        raise ValueError(
            f"{path}: manifest dataset is {m.get('dataset')!r}; C2Prompt is "
            "aligned on ImageNet-R (official CIFAR-100 path is dead code)")
    for key in ("seed", "protocol", "outer", "surrogate", "test_pool",
                "inner"):
        if key not in m:
            raise ValueError(f"{path}: manifest missing {key!r}")
    p = m["protocol"]
    if (p["client_num"], p["task_num"], p["nb_classes"]) != (5, 5, 200):
        raise ValueError(
            f"{path}: protocol {p} does not match the frozen C2Prompt "
            "alignment (5 clients x 5 tasks, 200 classes)")
    _STATE["manifest"] = m
    return m


def walk_imagenet_r(root, split):
    """(paths, targets) in the exact data.continual_datasets.Imagenet_R
    ordering: sorted folders (class id) then sorted files."""
    fpath = os.path.join(os.path.expanduser(root), split)
    if not os.path.isdir(fpath):
        raise RuntimeError(f"{fpath}: not a directory (pass --data-root "
                           "pointing at the imagenet-r folder that contains "
                           "train/ and test/)")
    folders = sorted(d for d in os.listdir(fpath)
                     if os.path.isdir(os.path.join(fpath, d)))
    data, targets = [], []
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(fpath, folder)
        for ims in sorted(os.listdir(folder_path)):
            data.append(os.path.join(folder_path, ims))
            targets.append(idx)
    return data, targets


def build_state(data_root):
    """Precompute per-(client, class) train indices and per-class test-pool
    indices from the manifest + our data layout."""
    m = _STATE["manifest"]
    data_root = os.path.abspath(os.path.expanduser(data_root))
    _, train_targets = walk_imagenet_r(data_root, "train")
    _, test_targets = walk_imagenet_r(data_root, "test")
    if len(set(train_targets)) != 200 or len(set(test_targets)) != 200:
        raise RuntimeError(f"{data_root}: expected 200 classes in both "
                           "train/ and test/")

    client_class_train_idx = {}
    for cid in range(5):
        per = {}
        for tid in range(5):
            outer_raw = m["outer"][str(cid)][str(tid)]["raw_indices"]
            mask = set(m["outer"][str(cid)][str(tid)]["class_mask"])
            for pos in m["inner"][str(cid)][str(tid)]["train"]:
                raw = int(outer_raw[int(pos)])
                label = int(train_targets[raw])
                if label not in mask:
                    raise RuntimeError(
                        f"manifest inconsistency: client {cid} task {tid} "
                        f"raw index {raw} has class {label} not in the "
                        f"task mask {sorted(mask)}")
                per.setdefault(label, []).append(raw)
        client_class_train_idx[cid] = per

    test_pool_by_class = {}
    for raw in m["test_pool"]["indices"]:
        test_pool_by_class.setdefault(int(test_targets[int(raw)]), []) \
            .append(int(raw))

    _STATE["client_class_train_idx"] = client_class_train_idx
    _STATE["test_pool_by_class"] = test_pool_by_class
    return train_targets, test_targets


# ---------------------------------------------------------------------------
# dataset-level patches (iIMAGENET_R / iDataset)
# ---------------------------------------------------------------------------

def patch_dataset():
    """Override the iDataset data-selection API on iIMAGENET_R. Must be
    called with the C2Prompt root on sys.path (imports iDOMAINNET)."""
    import numpy as np
    from iDOMAINNET import iIMAGENET_R

    def manifest_load(self):
        split = "train" if (self.train or self.validation) else "test"
        data, targets = walk_imagenet_r(self.root, split)
        self.data = data
        self.targets = targets

    def manifest_getTrainData(self, classes, exemplar_set,
                              exemplar_label_set, client_index,
                              classes_real=None, classes_proportion=None,
                              class_distribution_client_di=None,
                              exe_class=None):
        if classes_real is None:
            query = [int(c) for c in classes]
        else:
            query = [int(c) for c in classes_real]
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [np.asarray(exemplar) for exemplar in exemplar_set]
            labels = [np.full((datas[k].shape[0],), label)
                      for k, label in enumerate(exemplar_label_set)]
            keep = (exe_class if exe_class is not None
                    else exemplar_label_set)
            datas_p, labels_p = [], []
            for c in keep:
                if c in exemplar_label_set:
                    datas_p.append(datas[list(exemplar_label_set).index(c)])
                    labels_p.append(labels[list(exemplar_label_set).index(c)])
            datas, labels = datas_p, labels_p
        per_class = _STATE["client_class_train_idx"][int(client_index)]
        for label in query:
            raws = per_class.get(label, [])
            if raws:
                data = self.data[np.asarray(raws, dtype=np.int64)]
            else:
                data = np.empty(0, dtype=self.data.dtype)
            datas.append(data)
            if classes_real is not None:
                mapped = classes[list(classes_real).index(label)]
            else:
                mapped = label
            labels.append(np.full((data.shape[0],), mapped))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def manifest_get_image_class_dict(self, label, client_index,
                                      classes_proportion=None,
                                      exemplar_set=None):
        per_class = _STATE["client_class_train_idx"][int(client_index)]
        raws = per_class.get(int(label), [])
        datas = ([str(p) for p in self.data[np.asarray(raws, dtype=np.int64)]]
                 if raws else [])
        if exemplar_set is not None:
            datas = datas + list(exemplar_set)
        return datas

    def manifest_getTestData_hard(self, classes, classes_real):
        datas, labels = [], []
        for label in classes_real:
            raws = _STATE["test_pool_by_class"].get(int(label), [])
            if raws:
                data = self.data[np.asarray(raws, dtype=np.int64)]
            else:
                data = np.empty(0, dtype=self.data.dtype)
            datas.append(data)
            labels.append(np.full((data.shape[0],),
                                  classes[list(classes_real).index(label)]))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    iIMAGENET_R.load = manifest_load
    iIMAGENET_R.getTrainData = manifest_getTrainData
    iIMAGENET_R.get_image_class_dict = manifest_get_image_class_dict
    iIMAGENET_R.getTestData_hard = manifest_getTestData_hard
    return iIMAGENET_R


# ---------------------------------------------------------------------------
# main.py source-level surgical patches
# ---------------------------------------------------------------------------

SOURCE_PATCHES = [
    # (old, new, expected occurrence count)
    # 1. class_distribution_client: virtual 20-class blocks -> manifest masks
    ("task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 20, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 20)))",
     "task_list.append(list(_C2P_MANIFEST[\"outer\"][str(i)][str(j)][\"class_mask\"]))",
     1),
    # 2. class_distribution_client_real: disable the sim-resampling branch
    #    (unique via `args.sim, 21`; the DomainNet twin uses 36)
    ("""                if i * int(args.epochs_global / args.tasks_global) + j > 0:
                    similar_global_task_id = random.sample(list(range(i * int(args.epochs_global / args.tasks_global) + j)), 1)[0]
                    similar_level = random.sample(list(range(args.sim, 21)), 1)[0]""",
     """                if False:
                    similar_global_task_id = None
                    similar_level = 0""",
     1),
    # 3. class_distribution_client_real appends (both occurrences; the
    #    deeper-indented one is dead after patch 2) -> manifest masks
    ("task_list.append(list(random.sample(list(range(200)), 20)))",
     "task_list.append(list(_C2P_MANIFEST[\"outer\"][str(i)][str(j)][\"class_mask\"]))",
     2),
    # 4. class_distribution_client_proportion: 20% slices -> full range
    #    (getTrainData is overridden anyway; this is a safe passthrough)
    ("task_list.append([start, start + 0.2])",
     "task_list.append([0.0, 1.0])",
     1),
    # 5. fork_rng: GPU 3 does not exist on single-GPU hosts
    ("with torch.random.fork_rng(devices=[3]):",
     "with torch.random.fork_rng(devices=[]):",
     1),
    # 6. task progression: FedTA-aligned full progression (all clients
    #    advance) instead of the official 50% sampling
    ("            old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.5))",
     "            old_client_1 = [i for i in range(overall_client)]",
     1),
    # 7. remove the hardcoded ImageNet_R 2-client switches (keyed on
    #    ep_g//3, i.e. tasks_global=3; wrong under tasks_global=5 and
    #    incompatible with full progression)
    ("""            if args.dataset == 'ImageNet_R':
                if ep_g // 3 == 1:
                    old_client_1 = [1, 2]
                elif ep_g // 3 == 2:
                    old_client_1 = [0, 4]
                elif ep_g // 3 == 3:
                    old_client_1 = [1, 3]
                elif ep_g // 3 == 4:
                    old_client_1 = [0, 4]""",
     "            pass",
     1),
    # 10. metrics recorder: observe the native boundary eval (it already
    #     evaluates EVERY trained client-task on the manifest test pool)
    #     and record the R[after_task][test_task] entry per client
    ("            acc_global_list.append(acc_global)",
     "            acc_global_list.append(acc_global)\n"
     "            _C2P_RECORDER.record(int(client_index), "
     "int(ep_g // args.tasks_global), int(model_task_id), "
     "float(acc_global), phase=\"boundary\")",
     1),
    # 11. flush the recorder CSVs after each boundary eval block
    ("        del model_for_eval",
     "        del model_for_eval\n"
     "        _C2P_RECORDER.flush(int(args.tasks_global))",
     1),
]


def patch_main_source(src):
    for old, new, count in SOURCE_PATCHES:
        got = src.count(old)
        if got != count:
            raise RuntimeError(
                f"main.py source patch failed: expected {count} "
                f"occurrence(s) of {old[:60]!r}..., found {got}; re-audit "
                "the official repo before patching")
        src = src.replace(old, new)
    return src


# ---------------------------------------------------------------------------
# models_Cprompt/vit_coda_p_hard.py source-level surgical patches (official
# cadence latents; all are masked by the official ImageNet-R cadence
# epochs_global=15 / tasks_global=3, where every ep_g%3==0 round is a task
# boundary and ep_g never exceeds 14. Under the frozen 25/5 protocol they
# fire.)
# ---------------------------------------------------------------------------

HARD_SOURCE_PATCHES = [
    # (old, new, expected occurrence count)
    # 8. global_class_stats fill gate (updateweight_with_promptchoosing):
    #    one line, two cadence latents.
    #    (a) non-boundary %3 rounds: the fill block reads `std` left over
    #    from the class_features loop, but the calculate loop only runs on
    #    new_task rounds, so class_features is empty -> UnboundLocalError.
    #    `and class_features` guards it (boundary rounds keep the official
    #    semantics bit-for-bit; a skipped non-boundary round is an identity
    #    recompute over unchanged class_stats anyway, and
    #    global_class_stats persists across rounds).
    #    (b) task starts that are not %3 rounds (observed: round 5 of the
    #    formal run, KeyError 126 in train_class): the official 15/3
    #    cadence makes task_id = ep_g//3 change exactly on %3 rounds, so
    #    `ep_g % 3 == 0` accidentally encodes "task start". Under the
    #    frozen 25/5 cadence task starts are ep_g=5,10,15,20; at ep_g=5
    #    the fill is skipped while train_class (which runs EVERY round,
    #    `ep_g >= 0`) queries the new task's classes -> KeyError on the
    #    first new-class batch. `or new_task` restores the official
    #    invariant "stats are filled before train_class at every task
    #    start". Equivalence: officially {new_task rounds} ==
    #    {ep_g%3==0 rounds} == {0,3,6,9,12} and class_features is
    #    non-empty exactly on those rounds, so the patched condition has
    #    the identical truth table on the official cadence.
    ("        if ep_g % 3 == 0:",
     "        if (ep_g % 3 == 0 or new_task) and class_features:",
     1),
    # 9. task-similarity sharpening temperature t: the official ImageNet_R
    #    schedule {1 if r%3==0; 0.9 for r in {2,5,8,11,14}=r%3==2;
    #    0.95 for r in {1,4,7,10,13}=r%3==1; else None} is cadence-fitted
    #    to 15 rounds and returns None (TypeError at `weight_point ** t`)
    #    for ep_g>14. Replace with the equivalent closed form (identical
    #    for all ep_g in 0..14, well-defined beyond).
    ("t = 1 if ep_g % 3 == 0 else 0.9 if ep_g in {2, 5, 8, 11, 14} else 0.95 if ep_g <= 14 else None",
     "t = 1 if ep_g % 3 == 0 else 0.9 if ep_g % 3 == 2 else 0.95",
     1),
]


def patch_hard_source(src):
    for old, new, count in HARD_SOURCE_PATCHES:
        got = src.count(old)
        if got != count:
            raise RuntimeError(
                f"vit_coda_p_hard.py source patch failed: expected {count} "
                f"occurrence(s) of {old[:60]!r}..., found {got}; re-audit "
                "the official repo before patching")
        src = src.replace(old, new)
    return src


def install_patched_hard_module():
    """Execute the patched vit_coda_p_hard.py source and register it in
    sys.modules BEFORE main.py's `from models_Cprompt.vit_coda_p_hard
    import vit_pt_imnet_hard` runs, so the official file is never imported
    unpatched (the repo itself stays zero-edit on disk)."""
    root = str(C2P_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    path = os.path.join(root, "models_Cprompt", "vit_coda_p_hard.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    src = patch_hard_source(src)

    import models_Cprompt  # noqa: F401  (package __init__)
    name = "models_Cprompt.vit_coda_p_hard"
    mod = types.ModuleType(name)
    mod.__file__ = path
    mod.__package__ = "models_Cprompt"
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def install_nni_stub():
    if "nni" not in sys.modules:
        stub = types.ModuleType("nni")
        stub.get_next_parameter = lambda: {}
        stub.report_intermediate_result = lambda *a, **k: None
        stub.report_final_result = lambda *a, **k: None
        sys.modules["nni"] = stub


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def build_argv(data_root, forwarded, manifest):
    """C2Prompt option.py argv: frozen protocol values first, forwarded
    user flags last (argparse last-wins for overrides)."""
    argv = [
        "--dataset", "ImageNet_R",
        "--device", "0",
        "--img_size", "224",
        "--numclass", "200",
        "--num_classes", "200",
        "--class_per_task", "8",
        "--task_size", "8",
        "--num_clients", "5",
        "--local_clients", "5",
        "--tasks_global", "5",
        "--epochs_global", "25",
        "--method", "cprompt_c2prompt_v2",
        "--prompt_flag", "codap_2d_v2",
        "--prompt_param", "25", "10", "10", "8", "0", "0", "6", "10", "8",
        "--learning_rate", "0.005",
        "--batch_size", "64",
        "--epochs_local", "10",
        "--global_update_lr", "100",
        "--seed", str(manifest["seed"]),
        "--easy", "0",
        "--imbalance", "none",
        "--dataroot", str(Path(data_root).resolve()),
    ]
    return argv + list(forwarded)


def main():
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--manifest", required=True)
    cli.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"
                                                 / "imagenet-r"),
                     help="imagenet-r folder containing train/ and test/ "
                          "(our FedTA layout; forwarded to C2Prompt as "
                          "--dataroot)")
    cli.add_argument("--run-name", default="manifest_adapter",
                     help="recorder run dir name under output/ImageNet-R/"
                          "c2prompt/ (default manifest_adapter; use a "
                          "distinct name for smoke runs so their metrics "
                          "never mix with the formal run)")
    known, forwarded = cli.parse_known_args()

    m = load_manifest(known.manifest)
    build_state(known.data_root)

    recorder = metrics_recorder.MetricsRecorder(
        out_root=REPO_ROOT / "output", dataset="ImageNet-R",
        method="c2prompt", seed=int(m["seed"]), run_name=known.run_name,
        meta={"manifest": str(Path(known.manifest).resolve()),
              "adapter": "protocol_adapters/run_c2p.py",
              "eval_pool": "manifest test_pool via C2Prompt native "
                           "model_global_eval_hard (observation only, "
                           "no added eval)"})

    root = str(C2P_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    install_nni_stub()
    patch_dataset()
    install_patched_hard_module()  # before main.py's import of it

    src_path = os.path.join(root, "main.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    src = patch_main_source(src)

    mod = types.ModuleType("c2prompt_main")
    mod.__file__ = src_path
    mod.__dict__["_C2P_MANIFEST"] = m
    mod.__dict__["_C2P_RECORDER"] = recorder

    sys.argv = ["main.py"] + build_argv(known.data_root, forwarded, m)

    print("=" * 70)
    print("[protocol-adapter] C2Prompt run under the frozen split manifest")
    print(f"[protocol-adapter] manifest : {known.manifest}")
    print(f"[protocol-adapter] c2prompt : {root}")
    print(f"[protocol-adapter] data root: {known.data_root}")
    p = m["protocol"]
    for k in ("client_num", "task_num", "private_class_num",
              "surrogate_num", "nb_classes"):
        print(f"[protocol-adapter] {k:18s}= {p[k]}")
    print(f"[protocol-adapter] seed               = {m['seed']}")
    print(f"[protocol-adapter] metrics            = {recorder.run_dir}")
    print("[protocol-adapter] frozen: tasks_global=5 epochs_global=25 "
          "local_clients=5 numclass=200 class_per_task=8")
    print("[protocol-adapter] method-native: lr=0.005 batch=64 "
          "epochs_local=10 prompt_param=25,10,10,8,0,0,6,10,8")
    print("[protocol-adapter] applied: manifest class_distribution "
          "injection, iIMAGENET_R load/getTrainData/"
          "get_image_class_dict/getTestData_hard overrides (train = "
          "inner-70% of the frozen outer subsets; eval pool = manifest "
          "test_pool), fork_rng devices=[] (single GPU), FedTA-aligned "
          "full task progression, nni stub, vit_coda_p_hard cadence-latent "
          "bug fixes x3 (std guard; global_class_stats fill gate fires on "
          "every task start, not just ep_g%3==0 rounds; t temperature "
          "closed form)")
    print("=" * 70)

    # C2Prompt main.py is a script (module-level training loop); executing
    # the patched source IS the run.
    exec(compile(src, src_path, "exec"), mod.__dict__)


if __name__ == "__main__":
    main()
