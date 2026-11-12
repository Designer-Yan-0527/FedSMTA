"""HePCo (TMLR 2024, external_baselines/hepco-fed) training entry with the
frozen protocol manifest injected.

Frozen design (external-baseline fairness policy, same as FedMGP/C2Prompt/
FedProTIP):

  The official HePCo implementation is preserved verbatim; ONLY the
  dataset-partition interface is replaced so HePCo consumes the exact same
  client/task/class/sample assignment as FedSMTA. No file inside
  external_baselines/hepco-fed is ever edited; all patches are applied at
  runtime from this module.

Protocol reconciliation for a GLOBAL class-IL model (same core problem as
FedProTIP, solved with the same first-arrival scheme):

  HePCo's global model is a single ViT-B/16 (timm augreg weights, cached)
  with a FULL-WIDTH head (nn.Linear(768, num_classes) built once), FedAvg-
  aggregated (trainer.communicate). All class bookkeeping keys off
  `tasks_real` (the per-task class lists) and `class_mapping` (raw class ->
  head column). The adapter sets BOTH to the first-arrival space:

      tasks[t]      = arrival positions [K_{t-1}, K_t)   (raw ids in the
                      official code -- positions here; targets coming out
                      of __getitem__ are class_mapping[raw] = arrival
                      positions, so column selection via tasks_real,
                      add_valid_output_dim(inc[t]) and the CE target space
                      are all one consistent label space)
      class_mapping = raw class -> first-arrival position
      CIFAR-100 increments [32, 26, 14, 16, 12]; ImageNet-R [40] * 5.

  HePCo's own machinery then runs natively: task-id bootstrap prompt
  selection, generator/KD consolidation over label_counts (raw-id domain,
  translated through class_mapping exactly like the official flow), and the
  official per-round server evals.

  Rotated public classes (CIFAR): a client's task-t samples whose class
  arrived at an earlier GLOBAL task are dropped from local training
  (DualPrompt.update_model masks old columns with -inf and HePCo runs with
  memory=0 -- the official experiment scripts set --memory 0, i.e. no
  coreset replay -- so there is no legal consumption path for old-class
  samples). Same accounting as FedProTIP: drops land in
  {run_dir}/drops.json; paper note "uses first-arrival samples only; no
  replay".

Injection points (all fail-loud, seal-checked by gate_hepco.py):

  1. iIMAGENET_R.load -> replaced with a sorted walk of OUR
     local_datasets/imagenet-r/{train,test} layout (the official loader
     globs a flat {class}/*.jpg layout that does not exist here and splits
     via random.seed(0) shuffle + 80/20 -- unusable under the frozen
     domain). Keeps the original Trainer.__init__ alive on INR (its
     natively-built datasets are discarded by patch 3 anyway).
  2. wandb / fairscale import stubs (neither is installed; wandb is only
     init/log'd, fairscale's checkpoint_wrapper is imported by models/vit.py
     but never applied because ckpt_layer=0).
  3. Trainer.__init__ wrapper: after the official init (which runs
     natively and is then discarded), tasks/tasks_logits/num_tasks/
     task_names/max_task, server_config['tasks'/'tasks_real'/
     'prompt_param'] are reconciled to the arrival space; the 5 per-client
     train datasets, 5 per-client test datasets and the server test dataset
     are rebuilt as manifest-backed instances; the server learner is
     rebuilt from the reconciled config.
  4. ManifestDataset(dataloaders.IMBALANCECIFAR | IMBALANCEINR): __init__
     builds archive[t] straight from the manifest --
       train(c, t)    = inner.train(c,t) restricted to classes NEW at t
                        (first-arrival drops counted),
       test server(t) = manifest test_pool samples of the classes new at t,
       test client(c,t) = test_pool samples of client c's task-t class
                        mask (public classes that arrived earlier are
                        fine: their columns already exist).
     load_dataset(train=True) also fills label_counts in the OFFICIAL
     raw-class-id domain (consumed by train_generator /
     knowledge_distillation, translated via class_mapping there).
     gen_imbalanced_data / percent / cutoff / imbalance are bypassed
     entirely (the manifest IS the partition). __getitem__/__len__/
     append_coreset are inherited untouched.
  5. NormalNN.cuda / DualPrompt.cuda source patch `> 1` -> `>= 1`: the
     official 2-GPU scripts always run under DataParallel and
     train_generator/knowledge_distillation hard-code
     `self.server.model.module.*`; single-device DataParallel is a pure
     wrapper (identical numerics) that keeps those accesses working on one
     GPU. CPU mode (gpuid=[-1], used by the gate) never calls cuda().
  6. Trainer.train source patch: after the official per-round server
     evals, on the FINAL round of each task an RNG-neutral boundary eval
     records (client_id, after_task, test_task, accuracy) into the
     RunManager-format CSV tree via protocol_adapters/metrics_recorder.py:
     per (client c, test task k<=t) the manifest test_pool restricted to
     mask(c,k), logits masked to arrived arrival positions, shuffle=False,
     model.eval(), no_grad -- zero RNG consumption, zero training impact.

Protocol mapping (frozen, enforced against the manifest):
  5 clients x 5 tasks (n_clients=5 official; full participation -- the
  official loop trains every client every round), seed = manifest seed (42),
  n_rounds=5 FL rounds per task (frozen protocol; official script 10),
  schedule=[30] local epochs per round (frozen protocol; official 10),
  eval pool = manifest test_pool.
Method-native (official configs/cifar100_vit_prompt.yaml,
configs/imnet-r_clip_prompt.yaml + experiments/*-hepco1.sh):
  learner=prompt/L2P, prompt_param=[100,20,1,-1], weighted_l2p,
  --kl --hepco, lambda_KL=1, lambda_mse=0.1, noise_dimension=64,
  replay_ratio=0.5, memory=0 (no coreset), Adam lr=0.005 momentum=0.9,
  batch=64, ViT-B/16 augreg-pretrained, kd_epochs=200,
  generator_epochs=100, kd_lr=generator_lr=1e-4, temp=2, top_k=1.
Memory/replay column: none (memory=0; first-arrival drop documented).

Usage (from the FedSMTA repo root, FedSMTA conda env; GPU runs must be
executed outside the Trae sandbox -- CUDA init is blocked inside it):

  python protocol_adapters/run_hepco.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      [--data-root local_datasets] [--device cuda:0] \
      [--rounds 5] [--local-epochs 30] \
      [--limit-tasks N]   # SMOKE ONLY: truncate to the first N tasks

Output: output/{cifar100|ImageNet-R}/hepco/manifest_adapter/seed_{seed}/
  (metrics/ CSV tree + meta.json + drops.json)
"""

import argparse
import inspect
import json
import os
import random
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
HEPCO_ROOT = REPO_ROOT / "external_baselines" / "hepco-fed"

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import metrics_recorder  # noqa: E402

_STATE = {
    "manifest": None,
    "increments": None,      # new-class counts per task (arrival order)
    "arrival": None,         # raw class -> arrival position
    "new_at_task": None,     # t -> sorted list of new raw classes
    "train_raw": None,       # (c, t) -> inner.train raw trainset indices
    "test_pool_arr": None,   # sorted raw test indices of the eval pool
    "pool_data": None,       # pool-domain data (uint8 arrays / path arrays)
    "pool_y": None,          # pool-domain raw targets
    "pool_arrival": None,    # pool-domain arrival positions
    "drops": {},             # (c, t) -> number of dropped samples
    "data_root": None,
    "svd_rerouted": False,
    "recorder": None,
    "dataset_cls": None,     # ManifestDataset class for this dataset
    "class_mapping": None,   # raw class -> arrival position
}

METRIC_KEYS = ["acc", "lastacc", "time", "plastic", "til", "cka",
               "predisacc", "predislastacc"]
SAVE_KEYS = ["global", "pt", "pt-local"]


# ---------------------------------------------------------------------------
# sealed env bugfix: numpy LAPACK correctness self-test + scipy reroute
# (mirrors run_fedprotip.install_robust_svd; HePCo itself does not call
# np.linalg.svd, but the policy is uniform across all adapters)
# ---------------------------------------------------------------------------


def install_robust_svd():
    """On hosts where numpy's OpenBLAS LAPACK silently returns WRONG
    singular values (verified on this autodl box), reroute
    np.linalg.svd -> scipy.linalg.svd for this process only. Fail-loud
    self-test against torch CPU (MKL); healthy hosts are untouched."""
    rng = np.random.RandomState(0)
    import torch
    healthy = True
    diag = []
    for shape in ((300, 100), (1152, 512), (4608, 512)):
        a = rng.randn(*shape).astype(np.float32)
        try:
            s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
            ref = torch.linalg.svdvals(torch.from_numpy(a)).numpy()
            ok = bool(np.isfinite(s).all()
                      and abs(float(s.max()) - float(ref.max()))
                      <= 1e-2 * max(1.0, float(ref.max())))
            diag.append(f"{shape}:np={float(s.max()):.3f} torch="
                        f"{float(ref.max()):.3f}")
        except Exception:
            ok = False
            diag.append(f"{shape}:exception")
        healthy &= ok
    _STATE["svd_rerouted"] = not healthy
    if healthy:
        print("[protocol-adapter] numpy LAPACK self-test healthy "
              f"({'; '.join(diag)})")
        return
    print("[protocol-adapter] WARNING: numpy LAPACK self-test FAILED "
          f"({'; '.join(diag)}) -- rerouting np.linalg.svd "
          "to scipy.linalg.svd for this process (sealed env bugfix)")
    import scipy.linalg as sla

    def _robust_svd(a, full_matrices=True, compute_uv=True,
                    hermitian=False):
        if compute_uv:
            return sla.svd(a, full_matrices=full_matrices)
        return sla.svd(a, full_matrices=full_matrices, compute_uv=False)

    np.linalg.svd = _robust_svd


# ---------------------------------------------------------------------------
# manifest / arrival-order bookkeeping (same scheme as run_fedprotip)
# ---------------------------------------------------------------------------


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema") != "fedsmta-split-manifest-v1":
        raise ValueError(f"{path}: unsupported schema {m.get('schema')!r}")
    if m.get("dataset") not in ("cifar100", "ImageNet-R"):
        raise ValueError(f"{path}: dataset {m.get('dataset')!r} not "
                         "supported by the HePCo adapter")
    p = m["protocol"]
    if (p["client_num"], p["task_num"]) != (5, 5):
        raise ValueError(f"{path}: protocol {p} does not match the frozen "
                         "HePCo alignment (5 clients x 5 tasks)")
    _STATE["manifest"] = m
    return m


def compute_arrival():
    """First-arrival class order over the union of the 5 clients' masks
    (identical to run_fedprotip.compute_arrival)."""
    m = _STATE["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    new_at_task, seen = [], set()
    for t in range(nt):
        u = set()
        for c in range(nc):
            u |= set(m["outer"][str(c)][str(t)]["class_mask"])
        new = sorted(u - seen)
        seen |= set(new)
        new_at_task.append(new)
    total = m["protocol"]["nb_classes"]
    if len(seen) != total:
        raise RuntimeError(f"arrival order covers {len(seen)} of {total} "
                           "classes -- manifest is not a valid class cover")
    increments = [len(x) for x in new_at_task]
    arrival = {}
    pos = 0
    for t in range(nt):
        for cl in new_at_task[t]:
            arrival[cl] = pos
            pos += 1
    _STATE["increments"] = increments
    _STATE["arrival"] = arrival
    _STATE["new_at_task"] = new_at_task
    _STATE["class_mapping"] = dict(arrival)
    _STATE["class_mapping"][-1] = -1
    return increments, arrival, new_at_task


def build_train_raw():
    """(c, t) -> raw trainset indices of the manifest inner 70% split."""
    m = _STATE["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    tr = {}
    for c in range(nc):
        for t in range(nt):
            outer_raw = m["outer"][str(c)][str(t)]["raw_indices"]
            tr[(c, t)] = [int(outer_raw[int(i)])
                          for i in m["inner"][str(c)][str(t)]["train"]]
    _STATE["train_raw"] = tr
    return tr


def walk_imagenet_r(root, split):
    """(paths, targets) in the main-repo Imagenet_R ordering: sorted
    folders (class id) then sorted files == the manifest raw domain."""
    fpath = os.path.join(os.path.expanduser(root), split)
    if not os.path.isdir(fpath):
        raise RuntimeError(f"{fpath}: not a directory (pass --data-root "
                           "pointing at the imagenet-r folder that "
                           "contains train/ and test/)")
    folders = sorted(d for d in os.listdir(fpath)
                     if os.path.isdir(os.path.join(fpath, d)))
    data, targets = [], []
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(fpath, folder)
        for ims in sorted(os.listdir(folder_path)):
            data.append(os.path.join(folder_path, ims))
            targets.append(idx)
    return data, targets


# ---------------------------------------------------------------------------
# import stubs (wandb / fairscale are not installed in the frozen env)
# ---------------------------------------------------------------------------


def install_import_stubs():
    if "wandb" not in sys.modules:
        stub = types.ModuleType("wandb")

        def _noop(*a, **k):
            return None
        stub.init = _noop
        stub.log = _noop
        stub.finish = _noop
        stub.config = types.SimpleNamespace(update=_noop)
        sys.modules["wandb"] = stub
    # models/vit.py does `from fairscale.nn.checkpoint.checkpoint_activations
    # import checkpoint_wrapper` but builds VisionTransformer with
    # ckpt_layer=0 -> never applied; a passthrough keeps the import honest
    # without touching the env.
    if "fairscale" not in sys.modules:
        def _ckpt_wrapper(m, *a, **k):
            return m
        ca = types.ModuleType(
            "fairscale.nn.checkpoint.checkpoint_activations")
        ca.checkpoint_wrapper = _ckpt_wrapper
        ck = types.ModuleType("fairscale.nn.checkpoint")
        ck.checkpoint_activations = ca
        nn_mod = types.ModuleType("fairscale.nn")
        nn_mod.checkpoint = ck
        fs = types.ModuleType("fairscale")
        fs.nn = nn_mod
        sys.modules["fairscale"] = fs
        sys.modules["fairscale.nn"] = nn_mod
        sys.modules["fairscale.nn.checkpoint"] = ck
        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = ca
    os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# raw-domain pools (the manifest raw_indices/raw test domain)
# ---------------------------------------------------------------------------


def build_pools(data_root):
    dataset = _STATE["manifest"]["dataset"]
    root = os.path.abspath(os.path.expanduser(data_root))
    if dataset == "cifar100":
        from torchvision import datasets as tvd
        if not os.path.isdir(os.path.join(root, "cifar-100-python")):
            raise RuntimeError(f"{root}: cifar-100-python not found")
        tr = tvd.CIFAR100(root, train=True, download=False)
        te = tvd.CIFAR100(root, train=False, download=False)
        train_pool = (np.asarray(tr.data), np.array(tr.targets))
        test_all = (np.asarray(te.data), np.array(te.targets))
    else:
        inr = os.path.join(root, "imagenet-r")
        tr_d, tr_t = walk_imagenet_r(inr, "train")
        te_d, te_t = walk_imagenet_r(inr, "test")
        train_pool = (np.array(tr_d), np.array(tr_t))
        test_all = (np.array(te_d), np.array(te_t))
    _STATE["data_root"] = root
    _STATE["train_pool"] = train_pool

    m = _STATE["manifest"]
    pool = sorted(int(x) for x in m["test_pool"]["indices"])
    if pool and pool[-1] >= len(test_all[1]):
        raise RuntimeError("test_pool exceeds the raw test domain")
    arrival = _STATE["arrival"]
    pool_data = test_all[0][np.array(pool, dtype=np.int64)]
    pool_y = test_all[1][np.array(pool, dtype=np.int64)]
    _STATE["test_pool_arr"] = np.array(pool, dtype=np.int64)
    _STATE["pool_data"] = pool_data
    _STATE["pool_y"] = pool_y
    _STATE["pool_arrival"] = np.array([arrival[int(c)] for c in pool_y],
                                      dtype=np.int64)


# ---------------------------------------------------------------------------
# dataset-level patch 1: INR loader (keeps the official Trainer.__init__
# alive on ImageNet-R and pins the raw domain to our sorted walk)
# ---------------------------------------------------------------------------


def patch_imagenet_r_load():
    dataset = _STATE["manifest"]["dataset"]
    if dataset != "ImageNet-R":
        return
    import dataloaders as dl_mod
    official_src = inspect.getsource(dl_mod.iIMAGENET_R.load)
    if "0.8*n_data" not in official_src or "random.seed(0)" not in \
            official_src:
        raise RuntimeError("dataloaders/dataloader.py drift: "
                           "iIMAGENET_R.load no longer shuffles with an "
                           "80/20 split -- re-audit before patching")
    inr_root = os.path.join(_STATE["data_root"], "imagenet-r")

    def manifest_inr_load(self):
        split = "train" if (self.train or self.validation) else "test"
        d, t = walk_imagenet_r(inr_root, split)
        self.data, self.targets = d, t
    dl_mod.iIMAGENET_R.load = manifest_inr_load
    # IMBALANCEINR subclasses iIMAGENET_R -> inherits the patch


# ---------------------------------------------------------------------------
# dataset-level patch 2: the manifest-backed dataset class
# ---------------------------------------------------------------------------


def make_manifest_dataset_class(base_cls, is_cifar):
    """ManifestDataset(dataloaders.IMBALANCECIFAR|IMBALANCEINR): archive
    built from the manifest; load_dataset serves manifest semantics and
    fills label_counts in the official raw-class-id domain."""

    class ManifestDataset(base_cls):
        def __init__(self, transform, train, client_idx, seed,
                     tasks_pos, class_mapping):
            st = _STATE
            m = st["manifest"]
            inc = st["increments"]
            nt = len(inc)
            self.root = st["data_root"]
            self.transform = transform
            self.train = train
            self.validation = False
            self.seed = seed
            self.t = -1
            self.tasks = tasks_pos
            self.class_mapping = dict(class_mapping)
            self.class_mapping[-1] = -1
            self.download_flag = False
            self.rand_split = False
            self.client_idx = client_idx
            self.num_classes = m["protocol"]["nb_classes"]
            # unused by the manifest path (gen_imbalanced_data bypassed)
            self.imb_type = "exp"
            self.imb_factor = 1.0
            self.percent = 1.0

            X_tr, y_tr = st["train_pool"]
            pool_data = st["pool_data"]
            pool_y = st["pool_y"]
            pool_arrival = st["pool_arrival"]
            arrival = st["arrival"]

            self.archive = []
            if train:
                for t in range(nt):
                    k0 = int(sum(inc[:t]))
                    k1 = k0 + int(inc[t])
                    raws = st["train_raw"][(client_idx, t)]
                    keep = np.array(
                        [r for r in raws
                         if k0 <= arrival[int(y_tr[r])] < k1],
                        dtype=np.int64)
                    st["drops"][f"c{client_idx}t{t}"] = \
                        len(raws) - len(keep)
                    self.archive.append((
                        np.asarray(X_tr[keep]),
                        np.array([y_tr[r] for r in keep],
                                 dtype=np.int64)))
            else:
                for t in range(nt):
                    if client_idx < 0:  # server view: classes new at t
                        k0 = int(sum(inc[:t]))
                        k1 = k0 + int(inc[t])
                        sel = ((pool_arrival >= k0)
                               & (pool_arrival < k1))
                    else:  # client view: client's task-t class mask
                        mask = m["outer"][str(client_idx)][str(t)][
                            "class_mask"]
                        sel = np.isin(
                            pool_y,
                            np.array(sorted(int(x) for x in mask),
                                     dtype=np.int64))
                    self.archive.append((
                        np.asarray(pool_data[sel]),
                        np.asarray(pool_y[sel], dtype=np.int64)))

            self.data = np.zeros(0)
            self.targets = np.zeros(0)
            self.coreset = (np.zeros(0), np.zeros(0))

        def load_dataset(self, t, train=True, label_counts=None, seed=-1,
                         cutoff=False, cutoff_ratio=0):
            if train:
                self.data, self.targets = self.archive[t]
                self.t = t
                if (label_counts is not None and self.client_idx >= 0
                        and self.train):
                    counts = np.bincount(
                        np.asarray(self.targets, dtype=np.int64),
                        minlength=self.num_classes)
                    for j in range(self.num_classes):
                        label_counts[j] = int(counts[j])
                    return label_counts
            else:
                self.data = np.concatenate(
                    [self.archive[s][0] for s in range(t + 1)], axis=0)
                self.targets = np.concatenate(
                    [self.archive[s][1] for s in range(t + 1)], axis=0)
                self.t = t

    return ManifestDataset


# ---------------------------------------------------------------------------
# learner patch: DataParallel on a single GPU (official code hard-codes
# .module accesses; 2-GPU scripts always run under DataParallel)
# ---------------------------------------------------------------------------


def patch_learner_cuda():
    import learners.default as ldefault
    import learners.prompt as lprompt
    from learners.default import NormalNN
    from learners.prompt import DualPrompt
    for cls, mod in ((NormalNN, ldefault), (DualPrompt, lprompt)):
        src = textwrap.dedent(inspect.getsource(cls.cuda))
        old = "if len(self.config['gpuid']) > 1:"
        if src.count(old) != 1:
            raise RuntimeError(
                f"learners/{mod.__name__}.py drift: expected 1 "
                f"occurrence of the DataParallel guard in {cls.__name__}"
                ".cuda, found "
                f"{src.count(old)}; re-audit the official repo")
        src = src.replace(old, "if len(self.config['gpuid']) >= 1:")
        ns = dict(vars(mod))
        exec(compile(src, f"<{cls.__name__}.cuda manifest patch>", "exec"),
             ns)
        cls.cuda = ns["cuda"]


# ---------------------------------------------------------------------------
# learner patch 2 (sealed OFFICIAL-REPO BUGFIX): kd.LWF.learn_batch calls
# update_model(..., prev_server=...) but DualPrompt.update_model declares
# the same object as prev_task_model (renamed on one side -> TypeError on
# every training batch). prev_task_model is referenced ONLY inside the
# loss_type=='triplet' branch (never taken in the frozen run, loss_type
# =None); the alias restores the official intent with zero effect on
# the frozen path. Official file untouched; anchor sealed in the gate.
# ---------------------------------------------------------------------------

UPDATE_MODEL_SIG_ANCHOR = (
    "def update_model(self, inputs, targets, target_KD = None, "
    "loss_type = None, server_model = None, lambda_prox = 0.01,"
    "prev_task_model = None):")


def patch_dualprompt_update_model():
    import learners.prompt as lprompt
    from learners.prompt import DualPrompt
    src = textwrap.dedent(inspect.getsource(DualPrompt.update_model))
    if src.count(UPDATE_MODEL_SIG_ANCHOR) != 1:
        raise RuntimeError(
            "learners/prompt.py drift: expected the official "
            "DualPrompt.update_model signature "
            f"(count={src.count(UPDATE_MODEL_SIG_ANCHOR)}); re-audit "
            "the prev_server/prev_task_model mismatch before patching")
    new_sig = (
        "def update_model(self, inputs, targets, target_KD = None, "
        "loss_type = None, server_model = None, lambda_prox = 0.01, "
        "prev_task_model = None, prev_server = None):\n"
        "    if prev_task_model is None:\n"
        "        prev_task_model = prev_server")
    src = src.replace(UPDATE_MODEL_SIG_ANCHOR, new_sig)
    ns = dict(vars(lprompt))
    exec(compile(src, "<DualPrompt.update_model manifest bugfix>", "exec"),
         ns)
    DualPrompt.update_model = ns["update_model"]


# ---------------------------------------------------------------------------
# trainer patch 1: __init__ wrapper (reconcile tasks + rebuild datasets
# and the server learner on the manifest)
# ---------------------------------------------------------------------------

_ORIG_TRAINER_INIT = None


def patch_trainer_init():
    global _ORIG_TRAINER_INIT
    import trainer as trainer_mod
    from trainer import Trainer
    _ORIG_TRAINER_INIT = Trainer.__init__

    def manifest_trainer_init(self, args, seed, metric_keys, save_keys):
        # official init runs natively first (its task/dataset/server
        # objects are discarded by the reconciliation below)
        _ORIG_TRAINER_INIT(self, args, seed, metric_keys, save_keys)
        inc = _STATE["increments"]
        nt = len(inc)
        k0 = 0
        ranges = []
        for t in range(nt):
            ranges.append(list(range(k0, k0 + inc[t])))
            k0 += inc[t]
        self.tasks = ranges
        self.tasks_logits = [list(r) for r in ranges]
        self.num_tasks = nt
        self.task_names = [str(i + 1) for i in range(nt)]
        self.max_task = nt
        self.server_config["tasks"] = self.tasks_logits
        self.server_config["tasks_real"] = self.tasks
        self.server_config["prompt_param"] = [nt, args.prompt_param]

        import dataloaders.utils as dutils
        resize = args.model_name.startswith("vit")
        tr_tf = dutils.get_transform(dataset=args.dataset, phase="train",
                                     aug=args.train_aug,
                                     resize_imnet=resize)
        te_tf = dutils.get_transform(dataset=args.dataset, phase="test",
                                     aug=args.train_aug,
                                     resize_imnet=resize)
        cls = _STATE["dataset_cls"]
        cm = _STATE["class_mapping"]
        self.train_datasets = [
            cls(transform=tr_tf, train=True, client_idx=c, seed=seed,
                tasks_pos=ranges, class_mapping=cm)
            for c in range(self.n_clients)]
        self.test_datasets = [
            cls(transform=te_tf, train=False, client_idx=c, seed=seed,
                tasks_pos=ranges, class_mapping=cm)
            for c in range(self.n_clients)]
        self.server_test_dataset = cls(
            transform=te_tf, train=False, client_idx=-1, seed=seed,
            tasks_pos=ranges, class_mapping=cm)

        import learners
        self.server = learners.__dict__[self.learner_type].__dict__[
            self.learner_name](self.server_config)
        self.prev_server = None

    Trainer.__init__ = manifest_trainer_init


# ---------------------------------------------------------------------------
# trainer patch 2: train() source patch (boundary eval on the final round)
# ---------------------------------------------------------------------------


def _hepco_boundary_eval(trainer, i):
    """RNG-neutral boundary eval: the global (FedAvg) model on the
    manifest test_pool, per (client c, test task k<=i) restricted to
    client c's frozen task-k class mask, logits masked to arrived
    arrival positions. Pure observation; no RNG consumption."""
    import torch
    from torch.utils.data import DataLoader

    st = _STATE
    m = st["manifest"]
    nc = m["protocol"]["client_num"]
    nt = len(st["increments"])
    arrived = int(sum(st["increments"][:i + 1]))
    is_cifar = m["dataset"] == "cifar100"
    pool_data = st["pool_data"]
    pool_y = st["pool_y"]
    pool_arrival = st["pool_arrival"]
    transform = trainer.server_test_dataset.transform
    model = trainer.server.model
    model.eval()
    dev = next(model.parameters()).device
    bs = trainer.batch_size
    with torch.no_grad():
        for c in range(nc):
            for k in range(i + 1):
                mask = m["outer"][str(c)][str(k)]["class_mask"]
                sel = np.isin(
                    pool_y,
                    np.array(sorted(int(x) for x in mask),
                             dtype=np.int64))
                if not sel.any():
                    continue
                view = _BoundaryEvalView(pool_data[sel],
                                         pool_arrival[sel], transform,
                                         is_cifar)
                loader = DataLoader(view, batch_size=bs, shuffle=False,
                                    num_workers=0, drop_last=False)
                correct = total = 0
                for x, y, _t in loader:
                    x = x.to(dev)
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    pred = out[:, :arrived].argmax(dim=1).cpu().numpy()
                    y = y.numpy()
                    correct += int((pred == y).sum())
                    total += len(y)
                acc = 100.0 * correct / total if total else float("nan")
                st["recorder"].record(c, i, k, acc, phase="boundary")
    st["recorder"].flush(nt)


class _BoundaryEvalView(object):
    """Minimal (img, arrival_position_target, 0) view over a pool slice."""

    def __init__(self, data, targets, transform, is_cifar):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.is_cifar = is_cifar

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        from PIL import Image
        if self.is_cifar:
            img = Image.fromarray(self.data[idx])
        else:
            img = Image.open(str(self.data[idx])).convert("RGB")
        return (self.transform(img), int(self.targets[idx]), 0)


TRAIN_SOURCE_ANCHOR = ("server_temp_table['lastacc'].append("
                      "self.server_task_eval(i-1, all_tasks=True))")


def patch_trainer_train():
    import trainer as trainer_mod
    from trainer import Trainer
    src = textwrap.dedent(inspect.getsource(Trainer.train))
    got = src.count(TRAIN_SOURCE_ANCHOR)
    if got != 1:
        raise RuntimeError(
            f"trainer.py drift: expected 1 occurrence of the boundary-eval "
            f"anchor in Trainer.train, found {got}; re-audit the official "
            "repo before patching")
    repl = (TRAIN_SOURCE_ANCHOR
            + "\n\n            if r == self.n_rounds - 1:\n"
              "                _HEPCO_BOUNDARY_EVAL(self, i)")
    src = src.replace(TRAIN_SOURCE_ANCHOR, repl)
    ns = dict(vars(trainer_mod))
    ns["_HEPCO_BOUNDARY_EVAL"] = _hepco_boundary_eval
    ns["_HEPCO_STATE"] = _STATE
    exec(compile(src, "<Trainer.train manifest patch>", "exec"), ns)
    Trainer.train = ns["train"]


# ---------------------------------------------------------------------------
# args / config
# ---------------------------------------------------------------------------


def build_args(cli):
    """Official YAML + official experiments/*-hepco1.sh flags, then the
    frozen protocol overrides."""
    import yaml
    m = _STATE["manifest"]
    dataset = m["dataset"]
    seed = int(m["seed"])
    if int(cli.seed) != seed:
        raise SystemExit(f"--seed {cli.seed} does not match the manifest "
                         f"seed {seed}: run a matching manifest, not an "
                         "off-protocol seed")
    cfg_rel = ("configs/cifar100_vit_prompt.yaml" if dataset == "cifar100"
               else "configs/imnet-r_clip_prompt.yaml")
    with open(HEPCO_ROOT / cfg_rel, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.update({
        # official experiments/*-hepco1.sh flags (method-native)
        "gpuid": [int(cli.gpuid)],
        "log_dir": str(cli.log_dir),
        "learner_type": "prompt",
        "learner_name": "L2P",
        "debug_mode": 0,
        "repeat": 1,
        "overwrite": 1,
        "oracle_flag": False,
        "upper_bound_flag": False,
        "memory": 0,
        "temp": 2.0,
        "DW": False,
        "prompt_param": [100, 20, 1, -1],
        "mu": 0.0, "beta": 0.0, "eps": 0.0,
        "config": str(HEPCO_ROOT / cfg_rel),
        "template_style": "openai_imagenet_template",
        "freeze_last": False,
        "only_eval_zs": False,
        "n_clients": 5,
        "kl": True,
        "hepco": True,
        "imbalance": 1.0,
        "percent": 0.1,
        "cutoff": True,             # bypassed: the manifest IS the partition
        "cutoff_ratio": 0.4,
        "wandb_name": "protocol_adapter",
        "ignore_past_server": False,
        "lambda_KL": 1.0,
        "generator_epochs": 100,
        "generator_lr": 1e-4,
        "noise_dimension": 64,
        "kd_epochs": 200,
        "kd_lr": 1e-4,
        "replay_ratio": 0.5,
        "prompt_type": "weighted_l2p",
        "loss_type": None,
        "lambda_prox": 0.01,
        "lambda_mse": 0.1,
        # frozen protocol (hard)
        "seed": seed,
        "n_rounds": int(cli.rounds),        # 5 FL rounds per task
        "schedule": [int(cli.local_epochs)],  # 30 local epochs per round
        "max_task": len(_STATE["increments"]),
        "dataroot": str(Path(cli.data_root).expanduser()),
    })
    return argparse.Namespace(**config)


def main(argv=None):
    p = argparse.ArgumentParser(
        "HePCo + protocol adapter (frozen manifest split)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("--device", default="cuda:0",
                   help="'cuda:N' or 'cpu' (gpuid for the official code)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "output"))
    p.add_argument("--run-name", default="manifest_adapter")
    p.add_argument("--rounds", type=int, default=5,
                   help="FL rounds per task (frozen protocol: 5; "
                        "official script: 10)")
    p.add_argument("--local-epochs", type=int, default=30,
                   help="local epochs per round (frozen protocol: 30; "
                        "official config: 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-tasks", type=int, default=0,
                   help="SMOKE ONLY: truncate to the first N tasks "
                        "(off-protocol; formal runs must leave 0)")
    cli = p.parse_args(argv)
    if cli.device == "cpu":
        cli.gpuid = -1
    else:
        cli.gpuid = int(cli.device.split(":")[-1])

    install_import_stubs()
    install_robust_svd()
    load_manifest(cli.manifest)
    compute_arrival()
    if cli.limit_tasks:
        if not 0 < cli.limit_tasks < 5:
            raise SystemExit("--limit-tasks must be 1..4 (smoke only)")
        _STATE["increments"] = _STATE["increments"][:cli.limit_tasks]
    build_train_raw()
    build_pools(cli.data_root)

    # sys.path / cwd: hepco root first (trainer/dataloaders/learners/models
    # resolve there; imr_class_reverse_map.json + configs are relative)
    hepco_root = str(HEPCO_ROOT.resolve())
    if hepco_root not in sys.path:
        sys.path.insert(0, hepco_root)
    os.chdir(hepco_root)

    is_cifar = _STATE["manifest"]["dataset"] == "cifar100"
    patch_imagenet_r_load()
    import dataloaders as dl_mod
    base_cls = dl_mod.IMBALANCECIFAR if is_cifar else dl_mod.IMBALANCEINR
    _STATE["dataset_cls"] = make_manifest_dataset_class(base_cls, is_cifar)

    patch_learner_cuda()
    patch_dualprompt_update_model()
    patch_trainer_init()
    patch_trainer_train()

    recorder = metrics_recorder.MetricsRecorder(
        out_root=cli.out_root,
        dataset=_STATE["manifest"]["dataset"],
        method="hepco",
        seed=_STATE["manifest"]["seed"],
        run_name=cli.run_name,
        meta={
            "adapter": "run_hepco.py",
            "manifest": os.path.abspath(cli.manifest),
            "increments": _STATE["increments"],
            "new_at_task": _STATE["new_at_task"],
            "rounds_per_task": cli.rounds,
            "local_epochs": cli.local_epochs,
            "fracs": "full participation (all 5 clients every round)",
            "eval_pool": "manifest test_pool (surrogate-excluded)",
            "replay": "none (memory=0 official; first-arrival drop for "
                      "CIFAR rotated public classes; see drops.json)",
            "learner": "prompt/L2P weighted_l2p + hepco generator/KD",
            "numpy_lapack_selftest": (
                "BROKEN->scipy reroute" if _STATE["svd_rerouted"]
                else "healthy"),
            "limit_tasks": cli.limit_tasks or None,
        })
    _STATE["recorder"] = recorder
    cli.log_dir = recorder.run_dir

    args = build_args(cli)

    # mirror run.py's seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    import trainer as trainer_mod
    trainer = trainer_mod.Trainer(args, seed, METRIC_KEYS, SAVE_KEYS)
    print("[protocol-adapter] HePCo run under the frozen manifest")
    print(f"[protocol-adapter] increments (new classes/task) = "
          f"{_STATE['increments']}")
    print(f"[protocol-adapter] metrics = {recorder.run_dir}")
    t0 = time.time()
    trainer.train({})
    print(f"[protocol-adapter] train() done in "
          f"{(time.time() - t0) / 3600:.2f} h")

    drops = dict(sorted(_STATE["drops"].items()))
    with open(recorder.run_dir / "drops.json", "w",
              encoding="utf-8") as f:
        json.dump({"drops_per_client_task": drops,
                   "total_dropped": sum(drops.values())}, f, indent=2)
    print(f"[protocol-adapter] first-arrival drops: "
          f"{sum(drops.values())} samples -> drops.json")
    recorder.flush(len(_STATE["increments"]))


if __name__ == "__main__":
    main()
