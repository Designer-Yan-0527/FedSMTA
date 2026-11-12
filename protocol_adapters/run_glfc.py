"""GLFC (CVPR 2022, Dong et al., "Federated Class-Incremental Learning",
external_baselines/GLFC) training entry with the
frozen protocol manifest injected.

Frozen design (external-baseline fairness policy, same as FedMGP/C2Prompt/
FedProTIP/HePCo):

  The official GLFC implementation is preserved verbatim; ONLY the
  dataset-partition interface is replaced so GLFC consumes the exact same
  client/task/class/sample assignment as FedSMTA. No file inside
  external_baselines/GLFC is ever edited; all patches are applied at
  runtime from this module (fl_main.py is a flat SCRIPT -- the adapter
  reads its source, applies anchored fail-loud replacements and exec()s
  it, the same surgical-source-patch pattern HePCo used on Trainer.train).

Protocol reconciliation for a GLOBAL class-IL model (same core problem as
FedProTIP/HePCo, solved with the same FIRST-ARRIVAL scheme):

  GLFC's global model (resnet18_cbam + full-width fc) grows its head by
  `task_size` raw class-ids per task and every client samples 6 of the
  task's classes via random.sample over a GLOBAL id range -- it has no
  notion of per-client class ownership. The adapter reconciles this to
  the frozen manifest exactly like FedProTIP did:

      * class space = first-arrival order (CIFAR increments
        [32, 26, 14, 16, 12]; ImageNet-R [40] x 5);
      * numclass bookkeeping (beforeTrain) -> arrival cumulative K_t;
      * current_class -> the client's manifest mask(c, t) remapped to
        arrival positions (the manifest IS the partition; the official
        6-class random.sample is a partition mechanism and is replaced);
      * every label a model ever sees (one-hot target, exemplar label,
        test label) is an arrival position.

  Rotated public classes (CIFAR): GLFC is a REHEARSAL method -- its loss
  is full-width one-hot BCE with old-model distillation, old columns stay
  trainable, and every client keeps exemplar sets. A rotated public class
  whose global column already exists is therefore trained LEGALLY by the
  client that sees it again: zero drops (unlike FedProTIP/HePCo which
  have no legal consumption path). Each class still enters a client's
  exemplar set exactly once (every (client, class) pair appears in
  exactly one manifest task).

  Replay columns: exemplar memory (memory_size, official default 2000)
  + proxy-server gradient-inversion reconstruction are method-native
  and kept verbatim. Paper note for GLFC rows: "uses replay (exemplar
  memory + gradient-inversion rehearsal)".

ImageNet-R bridge (GLFC only ships CIFAR-100/Tiny/Mini loaders):

  The adapter pre-decodes local_datasets/imagenet-r/{train,test} (the
  main-repo Imagenet_R sorted-walk order == the manifest raw domain,
  same walk as run_fedprotip.walk_imagenet_r) and resizes every image to
  84 x 84 -- GLFC's own non-CIFAR idiom (its Mini_Imagenet branch runs on
  84x84 arrays with the same train/test transforms we keep verbatim:
  RandomCrop(img_size=32, padding=4) crops from 84x84, test Resize(32)).
  The whole pipeline therefore stays inside GLFC's native 32x32 design
  space: LeNet hideen=768, the (1,3,32,32) reconstruction dummy and the
  CIFAR normalization stats all remain untouched.

  Sealed hardcode bugfixes (INR only, numerically no-ops on CIFAR where
  nb_classes == 100): the label space of the tiny LeNet encoder and the
  proxy reconstruction loop are hardcoded to 100 classes --
  fl_main `LeNet(num_classes=100)`, ProxyServer.reconstruction
  `np.zeros((1, 100))` and `for label_i in range(100):` cannot represent
  ImageNet-R's 200 classes. Patched at runtime to the manifest
  nb_classes; gate_glfc.py PartC asserts all three anchors are still
  present in the official source.

  Sealed hardcode bugfix (both datasets): GLFC_model.train's LR schedule
  hardcodes the official local-epoch count (`ep_g * 20`, 3 occurrences)
  while epochs is a constructor arg. The frozen protocol runs 30 local
  epochs (as FedProTIP/HePCo do), so the hardcode is parameterized to
  `ep_g * self.epochs` (identical numerics at the official 20); the
  200-step decay period and boundary offsets stay verbatim.

Injection points (all fail-loud, seal-checked by gate_glfc.py):

  fl_main.py source patches (anchored, count-checked):
    P1 dataset branch      -> manifest-backed duck-type datasets
                              (ManifestClientDataset x5 + ManifestTestDataset)
    P2 client pool         -> `for i in range(125)` -> the manifest's 5
                              fixed clients, each owning ITS OWN dataset
                              object (official GLFC shares one global
                              trainset across all simulated clients; the
                              manifest is per-client, so each GLFC_model
                              gets its own client-scoped dataset)
    P3 client churn        -> neutralized (frozen: fixed 5-client cohort,
                              the official 90/10 churn + task_size new
                              clients per task is a partition mechanism)
    P4 head growth         -> classes_learned += arrival increments[t]
    P5 participation        -> all 5 clients every round (frozen frac=1.0;
                              official random.sample(local_clients=10))
    P6/P6b/P6c init widths -> network / proxyServer / classes_learned
                              start at K_0 = increments[0]
    P7 global eval call     -> _GLFC_GLOBAL_EVAL (official eval semantics
                              on the manifest test pool + RNG-neutral
                              boundary recording at each task boundary)
    P8 training_log         -> run_dir (the official ./training_log tree
                              is preserved verbatim inside it)
    P9 (INR only)           -> LeNet num_classes / reconstruction loops
                              -> manifest nb_classes
  GLFC.py runtime patches:
    beforeTrain            -> manifest version (numclass = arrival K_t,
                              current_class = mask(c,t) arrival positions;
                              official random.sample replaced)
    train                  -> source patch `ep_g * 20` ->
                              `ep_g * self.epochs` (3 anchors)
  ProxyServer.py runtime patch (INR only):
    reconstruction         -> 100-class hardcodes -> nb_classes

  cv2 is not installed in the frozen env and is imported (but never
  called) by mini_imagenet/proxy_data -> noop stub module, fail-loud on
  use. matplotlib backend -> Agg. wandb is not used by GLFC.

Protocol mapping (frozen, enforced against the manifest):
  5 clients x 5 tasks, full participation every round, seed = manifest
  seed (42), 5 FL rounds per task (epochs_global // tasks_global),
  30 local epochs per round (frozen protocol; official default 20),
  eval pool = manifest test_pool (surrogate-excluded).
Method-native (official option.py defaults, overridable):
  resnet18_cbam from scratch, SGD lr=2.0 wd=1e-5 with the native
  200-step decay schedule, batch_size=128, memory_size=2000,
  entropy-signal gating, prototype-mask gradient sharing, proxy-server
  DLG reconstruction (250 LBFGS iters x 20 images/class), exemplar
  herding (m = memory_size / learned classes).

Usage (from the FedSMTA repo root, FedSMTA conda env; GPU runs must be
executed OUTSIDE the sandbox -- CUDA init is blocked inside it):

  python protocol_adapters/run_glfc.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      [--data-root local_datasets] [--device 0] \
      [--rounds 5] [--epochs-local 30] [--batch-size 128] \
      [--memory-size 2000] [--limit-tasks N]   # SMOKE ONLY

Output: output/{cifar100|ImageNet-R}/glfc/manifest_adapter/seed_{seed}/
  (metrics/ CSV tree + meta.json + drops.json + log_tar_*.txt)
"""

import argparse
import inspect
import json
import os
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import metrics_recorder  # noqa: E402

GLFC_ROOT = REPO_ROOT / "external_baselines" / "GLFC" / "src"

_STATE = {
    "manifest": None,
    "increments": None,      # new-class counts per task (arrival order)
    "cumulative": None,      # cumulative head width K_t after task t
    "arrival": None,         # raw class -> arrival position
    "pos_to_raw": None,      # arrival position -> raw class
    "train_raw": None,       # (c, t) -> inner.train raw trainset indices
    "datasets": None,        # per-client ManifestClientDataset
    "test_dataset": None,    # ManifestTestDataset (test_pool backed)
    "recorder": None,
    "run_dir": None,
    "nb_classes": None,
    "n_tasks": None,         # effective task count (truncated on smoke)
    "base": None,            # raw arrays/targets of the train + test sets
    "svd_rerouted": False,
    "device": 0,
}

# ---------------------------------------------------------------------------
# sealed env bugfix: numpy LAPACK self-test + scipy reroute (see
# run_fedprotip.install_robust_svd; GLFC only needs np.linalg.norm on the
# hot path but the uniform env guard is cheap and catches a broken host)
# ---------------------------------------------------------------------------


def install_robust_svd():
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
# manifest / arrival-order bookkeeping (mirrors run_fedprotip.py)
# ---------------------------------------------------------------------------


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema") != "fedsmta-split-manifest-v1":
        raise ValueError(f"{path}: unsupported schema {m.get('schema')!r}")
    if m.get("dataset") not in ("cifar100", "ImageNet-R"):
        raise ValueError(f"{path}: dataset {m.get('dataset')!r} not "
                         "supported by the GLFC adapter")
    p = m["protocol"]
    if (p["client_num"], p["task_num"]) != (5, 5):
        raise ValueError(f"{path}: protocol {p} does not match the frozen "
                         "GLFC alignment (5 clients x 5 tasks)")
    _STATE["manifest"] = m
    _STATE["nb_classes"] = int(p["nb_classes"])
    return m


def compute_arrival():
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
    arrival, pos = {}, 0
    for t in range(nt):
        for cl in new_at_task[t]:
            arrival[cl] = pos
            pos += 1
    pos_to_raw = {v: k for k, v in arrival.items()}
    _STATE["increments"] = increments
    _STATE["cumulative"] = [int(sum(increments[:t + 1]))
                            for t in range(len(increments))]
    _STATE["arrival"] = arrival
    _STATE["pos_to_raw"] = pos_to_raw
    _STATE["new_at_task"] = new_at_task
    return increments, arrival, new_at_task


def build_train_raw():
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
# import stubs (cv2 is imported-but-unused on our paths; matplotlib Agg)
# ---------------------------------------------------------------------------


def install_import_stubs():
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "cv2" not in sys.modules:
        try:
            import cv2  # noqa: F401
        except ImportError:
            stub = types.ModuleType("cv2")

            def _fail(*a, **k):
                raise RuntimeError(
                    "cv2 is stubbed in the frozen env; this code path is "
                    "not exercised by the GLFC manifest adapter")

            stub.imread = _fail
            stub.cvtColor = _fail
            stub.COLOR_BGR2RGB = 4  # constant shim, never used
            sys.modules["cv2"] = stub
            print("[protocol-adapter] cv2 not installed -> noop stub "
                  "(GLFC's cv2 imports are dead on the manifest path)")


# ---------------------------------------------------------------------------
# raw base sets (the manifest raw_indices / test_pool domains)
# ---------------------------------------------------------------------------


def load_base(data_root):
    """Load the full train/test sets in the manifest's raw-index domain.

    cifar100   : torchvision CIFAR-100 order (same domain run_fedprotip
                 validated against the frozen manifests).
    ImageNet-R : main-repo Imagenet_R sorted-walk order, decoded and
                 resized to 84x84 uint8 once (GLFC's own non-CIFAR
                 idiom: its Mini_Imagenet branch consumes 84x84 arrays
                 under the verbatim 32-crop train / Resize(32) test
                 transforms).
    """
    dataset = _STATE["manifest"]["dataset"]
    root = os.path.expanduser(data_root)
    if dataset == "cifar100":
        if not os.path.isdir(os.path.join(root, "cifar-100-python")):
            raise RuntimeError(f"{root}: cifar-100-python not found")
        from torchvision.datasets import CIFAR100
        tr = CIFAR100(root=root, train=True, download=False)
        te = CIFAR100(root=root, train=False, download=False)
        _STATE["base"] = {
            "train_data": np.asarray(tr.data),
            "train_targets": np.asarray(tr.targets, dtype=np.int64),
            "test_data": np.asarray(te.data),
            "test_targets": np.asarray(te.targets, dtype=np.int64),
        }
    else:  # ImageNet-R
        from PIL import Image
        inr_root = os.path.join(root, "imagenet-r")

        def _decode(paths, targets):
            arr = np.zeros((len(paths), 84, 84, 3), dtype=np.uint8)
            for i, p in enumerate(paths):
                with Image.open(p) as im:
                    im = im.convert("RGB").resize((84, 84), Image.BILINEAR)
                    arr[i] = np.asarray(im, dtype=np.uint8)
            return arr, np.asarray(targets, dtype=np.int64)

        tr_p, tr_t = walk_imagenet_r(inr_root, "train")
        te_p, te_t = walk_imagenet_r(inr_root, "test")
        tr_d, tr_l = _decode(tr_p, tr_t)
        te_d, te_l = _decode(te_p, te_t)
        _STATE["base"] = {
            "train_data": tr_d, "train_targets": tr_l,
            "test_data": te_d, "test_targets": te_l,
        }
        print(f"[protocol-adapter] ImageNet-R decoded to 84x84: "
              f"train {tr_d.shape}, test {te_d.shape}")


# ---------------------------------------------------------------------------
# manifest-backed datasets (duck-type the official iCIFAR100 interface:
# TrainData/TrainLabels/TestData/TestLabels, getTrainData, getTestData,
# get_image_class, __getitem__, __len__)
# ---------------------------------------------------------------------------


class ManifestClientDataset:
    """One client's OWN slice of the frozen manifest (inner.train only),
    addressed in arrival-position label space."""

    def __init__(self, client_id):
        m = _STATE["manifest"]
        self.client_id = int(client_id)
        base = _STATE["base"]
        self._data = base["train_data"]
        self._targets = base["train_targets"]
        arrival = _STATE["arrival"]
        own = {}
        for t in range(m["protocol"]["task_num"]):
            for r in _STATE["train_raw"][(self.client_id, t)]:
                own.setdefault(int(self._targets[r]), []).append(int(r))
        self._own_by_class = {k: np.array(sorted(v), dtype=np.int64)
                              for k, v in own.items()}
        self._pos_to_raw = _STATE["pos_to_raw"]
        self.transform = None            # set by _GLFC_ATTACH_TRANSFORMS
        self.TrainData, self.TrainLabels = [], []

    # -- official interface -------------------------------------------------
    @staticmethod
    def _concatenate(datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        """classes are ARRIVAL POSITIONS (the adapter's label space).
        Mirrors iCIFAR100.getTrainData: exemplar replay first, then the
        current classes; the exemplar labels are already arrival
        positions because GLFC's learned_classes bookkeeping runs in
        that space."""
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label)
                      for label in exemplar_label_set]
        for label in classes:
            raw = self._pos_to_raw[int(label)]
            pos = self._own_by_class.get(int(raw))
            if pos is None or len(pos) == 0:
                raise RuntimeError(
                    f"manifest violation: client {self.client_id} owns no "
                    f"train samples of raw class {raw} (arrival position "
                    f"{label}) -- partition/label space mismatch")
            datas.append(self._data[pos])
            labels.append(np.full((len(pos)), label))
        self.TrainData, self.TrainLabels = self._concatenate(datas, labels)

    def get_image_class(self, label):
        """Arrival position -> the client's OWN inner.train arrays of that
        class (official: any client could read ALL samples of a class --
        the manifest restricts ownership to the client's own slice)."""
        raw = self._pos_to_raw[int(label)]
        pos = self._own_by_class.get(int(raw))
        if pos is None or len(pos) == 0:
            raise RuntimeError(
                f"get_image_class: client {self.client_id} owns no "
                f"samples of raw class {raw}")
        return self._data[pos]

    def getTrainItem(self, index):
        from PIL import Image
        img, target = Image.fromarray(self.TrainData[index]), \
            self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData):
            return self.getTrainItem(index)
        raise RuntimeError("client dataset has no TrainData (GLFC always "
                           "consumes the train branch)")

    def __len__(self):
        return len(self.TrainData)


class ManifestTestDataset:
    """The manifest test_pool (surrogate-excluded evaluation pool),
    addressed in arrival-position label space."""

    def __init__(self):
        base = _STATE["base"]
        pool = sorted(int(x) for x in
                      _STATE["manifest"]["test_pool"]["indices"])
        pool = np.array(pool, dtype=np.int64)
        if pool.max() >= len(base["test_targets"]):
            raise RuntimeError("test_pool exceeds the official test set")
        self._data = base["test_data"]
        self._pool = pool
        pool_targets = base["test_targets"][pool]
        by_class = {}
        for i, cl in enumerate(pool_targets):
            by_class.setdefault(int(cl), []).append(i)
        self._pool_by_class = {k: np.array(sorted(v), dtype=np.int64)
                               for k, v in by_class.items()}
        self._pos_to_raw = _STATE["pos_to_raw"]
        self.test_transform = None       # set by _GLFC_ATTACH_TRANSFORMS
        self.TestData, self.TestLabels = [], []

    @staticmethod
    def _concatenate(datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def _fill(self, positions):
        datas, labels = [], []
        for label in positions:
            raw = self._pos_to_raw[int(label)]
            pos = self._pool_by_class.get(int(raw))
            if pos is None or len(pos) == 0:
                raise RuntimeError(f"test pool has no samples of raw "
                                    f"class {raw} (position {label})")
            datas.append(self._data[self._pool[pos]])
            labels.append(np.full((len(pos)), label))
        self.TestData, self.TestLabels = self._concatenate(datas, labels)

    def getTestData(self, classes):
        """Official semantics: [classes[0], classes[1]) -- but in the
        arrival-position space (the adapter's reconciled label space)."""
        self._fill(list(range(int(classes[0]), int(classes[1]))))

    def subset(self, positions):
        """Arbitrary arrival-position subset (used by the RNG-neutral
        boundary eval; not part of the official interface)."""
        self._fill(sorted(set(int(p) for p in positions)))

    def getTestItem(self, index):
        from PIL import Image
        img, target = Image.fromarray(self.TestData[index]), \
            self.TestLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        return index, img, target

    def __getitem__(self, index):
        if self.TestData != []:
            return self.getTestItem(index)
        raise RuntimeError("test dataset has no TestData")

    def __len__(self):
        return len(self.TestData)


def build_datasets():
    m = _STATE["manifest"]
    _STATE["datasets"] = [ManifestClientDataset(c)
                          for c in range(m["protocol"]["client_num"])]
    _STATE["test_dataset"] = ManifestTestDataset()


# ---------------------------------------------------------------------------
# GLFC.py runtime patches: beforeTrain (manifest partition) + train
# (epochs hardcode seal)
# ---------------------------------------------------------------------------


def _manifest_before_train(self, task_id_new, group):
    """Replaces GLFC_model.beforeTrain: identical structure, but the
    class bookkeeping runs in the first-arrival space and the official
    `random.sample(range(numclass - task_size, numclass), 6)` partition
    mechanism becomes the client's frozen manifest mask.

    Zero-sample starvation: a mask class may own ZERO inner.train
    samples for this client (verified on the frozen CIFAR manifest:
    (c0,t4,raw 10) has its single outer sample in inner.test and
    (c2,t3,raw 56) has no outer sample at all). No method can train on
    zero samples, and GLFC's exemplar construction would crash on an
    empty class, so such classes are omitted from current_class and
    accounted in _STATE['omissions'] (-> drops.json). Their columns are
    still learned at their arrival task by the owning client."""
    c = self.train_dataset.client_id
    if task_id_new != self.task_id_old:
        self.task_id_old = task_id_new
        self.numclass = _STATE["cumulative"][task_id_new]
        if group != 0:
            if self.current_class != None:
                self.last_class = self.current_class
            mask = _STATE["manifest"]["outer"][str(c)][str(task_id_new)][
                "class_mask"]
            own = self.train_dataset._own_by_class
            kept, omitted = [], []
            for cl in sorted(int(x) for x in mask):
                pos = _STATE["arrival"][cl]
                if len(own.get(cl, [])) > 0:
                    kept.append(pos)
                else:
                    omitted.append(pos)
            self.current_class = kept
            if omitted:
                om = _STATE.setdefault("omissions", {})
                om[f"c{c}t{task_id_new}"] = omitted
        else:
            self.last_class = None

    self.train_loader = self._get_train_and_test_dataloader(
        self.current_class, False)


def patch_glfc_module():
    """import GLFC (official src on sys.path) and patch the class."""
    sys.path.insert(0, str(GLFC_ROOT))
    import GLFC as glfc_mod

    # seal check: the official partition mechanism must still be there
    src_bt = inspect.getsource(glfc_mod.GLFC_model.beforeTrain)
    if "random.sample([x for x in range(self.numclass - self.task_size, " \
       "self.numclass)], 6)" not in src_bt:
        raise RuntimeError("GLFC.py drift: beforeTrain no longer contains "
                           "the official random.sample partition -- "
                           "re-audit before patching")
    glfc_mod.GLFC_model.beforeTrain = _manifest_before_train

    # train: `ep_g * 20` hardcode -> `ep_g * self.epochs` (3 anchors)
    src = textwrap.dedent(inspect.getsource(glfc_mod.GLFC_model.train))
    n = src.count("ep_g * 20")
    if n != 3:
        raise RuntimeError(f"GLFC.py drift: expected 3 occurrences of the "
                           f"'ep_g * 20' local-epoch hardcode in "
                           f"GLFC_model.train, found {n} -- re-audit the "
                           "official repo before patching")
    src = src.replace("ep_g * 20", "ep_g * self.epochs")
    _STATE["train_patch_src"] = src
    ns = dict(vars(glfc_mod))
    ns["_GLFC_STATE"] = _STATE
    exec(compile(src, "<GLFC_model.train manifest patch>", "exec"), ns)
    glfc_mod.GLFC_model.train = ns["train"]
    # sealed environment fix: the official train_loader forks 8
    # DataLoader workers. The parent process has already initialised
    # CUDA by then (entropy_signal / model_to_device), and forked
    # workers deadlock probabilistically against the parent CUDA/pin
    # threads in this container (observed: 4/5 clients pass, the 5th
    # hangs forever inside the loader iterator with no traceback).
    # num_workers=0 loads data in-process -- numerically identical.
    src_l = textwrap.dedent(inspect.getsource(
        glfc_mod.GLFC_model._get_train_and_test_dataloader))
    if src_l.count("num_workers=8,") != 1:
        raise RuntimeError("GLFC.py drift: expected exactly 1 "
                           "'num_workers=8,' in "
                           "GLFC_model._get_train_and_test_dataloader -- "
                           "re-audit before patching")
    src_l = src_l.replace("num_workers=8,", "num_workers=0,")
    ns_l = dict(vars(glfc_mod))
    ns_l["_GLFC_STATE"] = _STATE
    exec(compile(src_l,
                 "<GLFC_model._get_train_and_test_dataloader patch>",
                 "exec"), ns_l)
    glfc_mod.GLFC_model._get_train_and_test_dataloader = \
        ns_l["_get_train_and_test_dataloader"]
    return glfc_mod


def patch_proxy_server(nb_classes):
    """Two sealed patches on proxyServer.reconstruction, applied for every
    dataset (textually identical on CIFAR where nb_classes == 100):

    1. INR hardcode bugfix: the proxy label space is hardcoded to 100
       classes (np.zeros((1, 100)), range(100)) which cannot represent
       ImageNet-R's 200 classes.
    2. sealed C' device reroute (user-approved 2026-11-12): the DLG
       inversion loop is a batch-1 LeNet double-backward -- kernel-launch
       bound; measured on this host 282ms/closure on the (shared) GPU vs
       13.2ms on CPU (21x). The inversion is rerouted to CPU: identical
       arithmetic, zero truncation, zero iteration reduction; the
       official loop, counts and Iteration stay verbatim. Only the
       privacy monitor's device placement changes; training, distillation
       and aggregation stay on GPU."""
    import ProxyServer as proxy_mod
    src = textwrap.dedent(
        inspect.getsource(proxy_mod.proxyServer.reconstruction))
    anchors = [("class_ratio = np.zeros((1, 100))",
                "class_ratio = np.zeros((1, _GLFC_STATE['nb_classes']))"),
               ("for label_i in range(100):",
                "for label_i in range(_GLFC_STATE['nb_classes']):")]
    for old, new in anchors:
        if src.count(old) != 1:
            raise RuntimeError(f"ProxyServer.py drift: expected exactly 1 "
                               f"occurrence of {old[:40]!r} in "
                               "proxyServer.reconstruction -- re-audit "
                               "before patching")
        src = src.replace(old, new)

    # sealed C' device reroute (both datasets): CPU executes this
    # launch-bound workload ~21x faster on this host; zero
    # method-semantic change (see docstring). Each anchor is
    # count-checked in the method source; after all five replacements
    # no '.to(self.device)' remains inside reconstruction.
    dev_anchors = [
        ("dummy_data = torch.randn((1, 3, 32, 32)).to(self.device)"
         ".requires_grad_(True)",
         "dummy_data = torch.randn((1, 3, 32, 32))"
         ".requires_grad_(True)"),
        ("label_pred = torch.Tensor([label_i]).long().to(self.device)"
         ".requires_grad_(False)",
         "label_pred = torch.Tensor([label_i]).long()"
         ".requires_grad_(False)"),
        ("criterion = nn.CrossEntropyLoss().to(self.device)",
         "criterion = nn.CrossEntropyLoss()"),
        ("recon_model = model_to_device(recon_model, False, self.device)",
         "recon_model = recon_model.cpu()"),
        ("grad_truth_temp = self.pool_grad[grad_index[0][j]]",
         "grad_truth_temp = [g.detach().cpu() for g in "
         "self.pool_grad[grad_index[0][j]]]"),
    ]
    for old, new in dev_anchors:
        if src.count(old) != 1:
            raise RuntimeError(f"ProxyServer.py drift: expected exactly 1 "
                               f"occurrence of {old[:48]!r} in "
                               "proxyServer.reconstruction -- re-audit "
                               "before patching")
        src = src.replace(old, new)
    _STATE["proxy_patch_src"] = src
    ns = dict(vars(proxy_mod))
    ns["_GLFC_STATE"] = _STATE
    exec(compile(src, "<proxyServer.reconstruction manifest patch>",
                 "exec"), ns)
    proxy_mod.proxyServer.reconstruction = ns["reconstruction"]


# ---------------------------------------------------------------------------
# fl_main.py source patches (anchored, count-checked) + exec
# ---------------------------------------------------------------------------

FL_MAIN_PATCHES = [
    # P1: dataset branch -> manifest-backed datasets
    ("""if args.dataset == 'cifar100':
    train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

elif args.dataset == 'tiny_imagenet':
    train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset

else:
    train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset""",
     """_GLFC_ATTACH_TRANSFORMS(train_transform, test_transform)
train_dataset = _GLFC_STATE['datasets'][0]
test_dataset = _GLFC_STATE['test_dataset']""", 1),
    # P2: client pool -> the manifest's 5 fixed clients, each with ITS OWN
    # (client-scoped) dataset; initial head width = K_0
    ("""for i in range(125):
    model_temp = GLFC_model(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                 args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model)
    models.append(model_temp)""",
     """for i in range(len(_GLFC_STATE['datasets'])):
    model_temp = GLFC_model(_GLFC_STATE['cumulative'][0], feature_extractor, args.batch_size, _GLFC_STATE['increments'][0], args.memory_size,
                 args.epochs_local, args.learning_rate, _GLFC_STATE['datasets'][i], args.device, encode_model)
    models.append(model_temp)""", 1),
    # P3: client churn -> frozen fixed 5-client cohort
    ("""    if task_id != old_task_id and old_task_id != -1:
        overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
        new_client = [i for i in range(overall_client, overall_client + args.task_size)]
        old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
        print(old_client_0)""",
     """    if task_id != old_task_id and old_task_id != -1:
        pass  # [protocol-adapter] frozen: fixed 5-client cohort, no churn""", 1),
    # P4: head growth follows the arrival increments
    ("        classes_learned += args.task_size",
     "        classes_learned += _GLFC_STATE['increments'][task_id]", 1),
    # P5: full participation (frozen frac=1.0)
    ("    clients_index = random.sample(range(num_clients), args.local_clients)",
     "    clients_index = list(range(num_clients))", 1),
    # P6: initial global head width = K_0
    ("model_g = network(args.numclass, feature_extractor)",
     "model_g = network(_GLFC_STATE['cumulative'][0], feature_extractor)", 1),
    # P6b: proxy server initial head width = K_0
    ("proxy_server = proxyServer(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform)",
     "proxy_server = proxyServer(args.device, args.learning_rate, _GLFC_STATE['cumulative'][0], feature_extractor, encode_model, train_transform)", 1),
    # P6c: classes_learned starts at K_0
    ("classes_learned = args.task_size",
     "classes_learned = _GLFC_STATE['cumulative'][0]", 1),
    # P7: global eval -> official semantics on the manifest test pool +
    # RNG-neutral boundary recording at each task boundary
    ("    acc_global = model_global_eval(model_g, test_dataset, task_id, args.task_size, args.device)",
     "    acc_global = _GLFC_GLOBAL_EVAL(model_g, test_dataset, task_id, ep_g, args.tasks_global, args.device)", 1),
    # P8: training_log -> run_dir (official tree preserved inside it)
    ("output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))",
     "output_dir = str(_GLFC_STATE['run_dir'])", 1),
]

FL_MAIN_PATCHES_INR = [
    # P9: LeNet encoder label space is hardcoded to 100 classes
    ("encode_model = LeNet(num_classes=100)",
     "encode_model = LeNet(num_classes=_GLFC_STATE['nb_classes'])", 1),
]


def patch_fl_main_source(nb_classes):
    src_path = GLFC_ROOT / "fl_main.py"
    src = src_path.read_text(encoding="utf-8")
    patches = list(FL_MAIN_PATCHES)
    if nb_classes != 100:
        patches += FL_MAIN_PATCHES_INR
    for old, new, count in patches:
        got = src.count(old)
        if got != count:
            raise RuntimeError(
                f"fl_main.py source patch failed: expected {count} "
                f"occurrence(s) of {old[:60]!r}..., found {got}; "
                "re-audit the official repo before patching")
        src = src.replace(old, new)
    return src


# ---------------------------------------------------------------------------
# evaluation: official global eval semantics + RNG-neutral boundary records
# ---------------------------------------------------------------------------


def _glfc_attach_transforms(train_transform, test_transform):
    for ds in _STATE["datasets"]:
        ds.transform = train_transform
    _STATE["test_dataset"].test_transform = test_transform


def _glfc_global_eval(model_g, test_dataset, task_id, ep_g, rounds_per_task,
                      device):
    """Official model_global_eval semantics, transplanted onto the
    manifest test pool (arrival-position labels, head width == K_t so no
    logit truncation is needed), + the protocol's RNG-neutral per-client
    boundary records on the FINAL round of each task (shuffle=False,
    no_grad, model.eval(): zero RNG consumption, zero training impact)."""
    import torch
    from torch.utils.data import DataLoader

    model_g.eval()
    k_arrived = _STATE["cumulative"][task_id]
    test_dataset.getTestData([0, k_arrived])
    loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    correct, total = 0, 0
    for _, imgs, labels in loader:
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    model_g.train()

    if ep_g % rounds_per_task == rounds_per_task - 1:
        _glfc_boundary_eval(model_g, test_dataset, task_id, device)
        _STATE["recorder"].flush(_STATE["n_tasks"])
    return float(accuracy)


def _glfc_boundary_eval(model_g, test_dataset, task_id, device):
    """Per (client c, test task k <= task_id): the manifest test pool
    restricted to client c's frozen task-k class mask (arrival-space
    labels). Mirrors run_fedprotip._fpt_boundary_eval."""
    import torch
    from torch.utils.data import DataLoader

    m = _STATE["manifest"]
    nc = m["protocol"]["client_num"]
    model_g.eval()
    with torch.no_grad():
        for c in range(nc):
            for k in range(task_id + 1):
                mask = m["outer"][str(c)][str(k)]["class_mask"]
                positions = sorted(_STATE["arrival"][int(cl)]
                                   for cl in mask)
                test_dataset.subset(positions)
                loader = DataLoader(test_dataset, batch_size=128,
                                    shuffle=False)
                correct, total = 0, 0
                for _, x, y in loader:
                    x, y = x.cuda(device), y.cuda(device)
                    out = model_g(x)
                    pred = out.argmax(dim=1).cpu()
                    correct += int((pred.numpy() == y.cpu().numpy()).sum())
                    total += len(y)
                if total:
                    acc = 100.0 * correct / total
                    _STATE["recorder"].record(c, task_id, k, acc,
                                              phase="boundary")
    model_g.train()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        "GLFC + protocol adapter (frozen manifest split)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("--device", type=int, default=0,
                   help="GPU ID (GLFC uses cuda:{device}; -1 unsupported "
                        "by the official model_to_device)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "output"))
    p.add_argument("--run-name", default="manifest_adapter")
    p.add_argument("--rounds", type=int, default=5,
                   help="FL rounds per task (frozen protocol: 5; official "
                        "10 via epochs_global=100/tasks_global=10)")
    p.add_argument("--epochs-local", type=int, default=30,
                   help="local epochs per round (frozen protocol: 30; "
                        "official default 20, hardcoded in its LR "
                        "schedule and sealed-patched to self.epochs)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="official GLFC default: 128")
    p.add_argument("--memory-size", type=int, default=2000,
                   help="exemplar memory budget (official default: 2000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-tasks", type=int, default=0,
                   help="SMOKE ONLY: truncate to the first N tasks "
                        "(off-protocol; formal runs must leave 0)")
    args_cli = p.parse_args(argv)

    install_import_stubs()
    install_robust_svd()
    load_manifest(args_cli.manifest)
    compute_arrival()
    if args_cli.seed != int(_STATE["manifest"]["seed"]):
        raise SystemExit(f"--seed {args_cli.seed} does not match the "
                         f"manifest seed {_STATE['manifest']['seed']}: "
                         "run a matching manifest, not an off-protocol "
                         "seed")
    if args_cli.limit_tasks:
        if not 0 < args_cli.limit_tasks < 5:
            raise SystemExit("--limit-tasks must be 1..4 (smoke only)")
        inc = _STATE["increments"][:args_cli.limit_tasks]
        _STATE["increments"] = inc
        _STATE["cumulative"] = [int(sum(inc[:t + 1]))
                                for t in range(len(inc))]
    _STATE["n_tasks"] = len(_STATE["increments"])
    build_train_raw()
    load_base(args_cli.data_root)
    build_datasets()
    _STATE["device"] = args_cli.device

    rounds, n_tasks = args_cli.rounds, _STATE["n_tasks"]
    nb = _STATE["nb_classes"]
    inc0 = _STATE["increments"][0]

    recorder = metrics_recorder.MetricsRecorder(
        out_root=args_cli.out_root,
        dataset=_STATE["manifest"]["dataset"],
        method="glfc",
        seed=_STATE["manifest"]["seed"],
        run_name=args_cli.run_name,
        meta={
            "adapter": "run_glfc.py",
            "manifest": os.path.abspath(args_cli.manifest),
            "increments": _STATE["increments"],
            "new_at_task": _STATE["new_at_task"],
            "rounds_per_task": rounds,
            "local_epochs": args_cli.epochs_local,
            "batch_size": args_cli.batch_size,
            "memory_size": args_cli.memory_size,
            "participation": "full (frozen frac=1.0; 5 fixed clients)",
            "eval_pool": "manifest test_pool (surrogate-excluded)",
            "replay": "method-native: exemplar memory + proxy-server "
                      "gradient-inversion rehearsal (uses replay)",
            "drops": "none (GLFC's full-width BCE + distillation consumes "
                     "rotated public-class samples legally)",
            "inr_bridge": ("pre-decoded to 84x84, GLFC Mini_Imagenet idiom; "
                           "official 32-crop/Resize(32) transforms kept "
                           "verbatim" if _STATE["manifest"]["dataset"] !=
                           "cifar100" else None),
            "numpy_lapack_selftest": (
                "BROKEN->scipy reroute" if _STATE["svd_rerouted"]
                else "healthy"),
            "limit_tasks": args_cli.limit_tasks or None,
            "proxy_recon_device": (
                "cpu (C' seal: batch-1 LeNet double-backward is "
                "launch-bound -- 282ms/closure GPU vs 13.2ms CPU on this "
                "host; arithmetic identical, no truncation, no iteration "
                "reduction)"),
        })
    _STATE["recorder"] = recorder
    _STATE["run_dir"] = recorder.run_dir

    # GLFC runtime patches (class-level), BEFORE fl_main's
    # `from GLFC import GLFC_model` picks the class up
    patch_glfc_module()
    patch_proxy_server(nb)

    # argv for the official option.args_parser
    sys.argv = [
        "fl_main.py",
        "--dataset", "cifar100",
        "--method", "glfc",
        "--numclass", str(inc0),
        "--img_size", "32",
        "--device", str(args_cli.device),
        "--batch_size", str(args_cli.batch_size),
        "--task_size", str(inc0),
        "--seed", str(args_cli.seed),
        "--memory_size", str(args_cli.memory_size),
        "--epochs_local", str(args_cli.epochs_local),
        "--learning_rate", "2.0",
        "--num_clients", "5",
        "--local_clients", "5",
        "--epochs_global", str(rounds * n_tasks),
        "--tasks_global", str(rounds),
    ]

    src = patch_fl_main_source(nb)
    os.chdir(str(GLFC_ROOT))
    ns = {
        "__name__": "__main__",
        "__file__": str(GLFC_ROOT / "fl_main.py"),
        "_GLFC_STATE": _STATE,
        "_GLFC_GLOBAL_EVAL": _glfc_global_eval,
        "_GLFC_ATTACH_TRANSFORMS": _glfc_attach_transforms,
    }
    print("[protocol-adapter] GLFC run under the frozen manifest")
    print(f"[protocol-adapter] increments (new classes/task) = "
          f"{_STATE['increments']}")
    print(f"[protocol-adapter] metrics = {recorder.run_dir}")
    t0 = time.time()
    exec(compile(src, str(GLFC_ROOT / "fl_main.py"), "exec"), ns)
    print(f"[protocol-adapter] fl_main finished in "
          f"{(time.time() - t0) / 3600:.2f} h")

    drops = {f"c{c}t{t}": 0
             for c in range(5) for t in range(n_tasks)}
    omissions = _STATE.get("omissions", {})
    with open(recorder.run_dir / "drops.json", "w",
              encoding="utf-8") as f:
        json.dump({"drops_per_client_task": drops,
                   "total_dropped": 0,
                   "zero_train_sample_classes_omitted": omissions,
                   "note": "GLFC is a rehearsal method: rotated "
                           "public-class samples are consumed legally "
                           "(existing columns stay trainable + exemplar "
                           "replay); no first-arrival drops. Mask "
                           "classes owning zero inner.train samples for "
                           "a client are untrainable by any method and "
                           "are omitted from that client's current_class "
                           "(arrival positions listed above); their "
                           "columns are learned at their arrival task "
                           "by the owning client."},
                  f, indent=2)
    recorder.flush(n_tasks)


if __name__ == "__main__":
    main()
