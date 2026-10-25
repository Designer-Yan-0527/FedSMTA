"""FedMGP protocol adapter: manifest-driven data partition injection.

Frozen design (external-baseline fairness policy):

  The official FedMGP implementation is preserved verbatim; ONLY the
  dataset-partition interface is replaced so FedMGP consumes the exact
  same client/task/class/sample assignment as FedSMTA. Paper wording:
  "We preserve the official implementation of each baseline and replace
  only the dataset partition interface to ensure identical client-task
  sample assignments."

All adapter code lives in the main repo (external_baselines/ is
gitignored); no file inside external_baselines/FedMGP is ever edited.

Three injection points (cifar100; the ImageNet-R section at the bottom
mirrors them on the main-repo lineage ImagenetR_spliter):

  1. cifar100_Data_Spliter.random_split()     -> outer split from manifest
  2. cifar100_Data_Spliter.process_testdata() -> surrogate / test pool
  3. torch.utils.data.random_split()          -> inner 70/30 from manifest

(3) also pins the EVALUATION sets: the inner 30% subsets are what both
codebases actually evaluate on (Client_DF.get_data -> random_split ->
test_loader). A same-seed native FedMGP run cannot reproduce them because
its torch RNG stream differs from FedSMTA's after model initialization,
even when the outer split is identical.

Import-order requirement: patch_torch_random_split() MUST be called
BEFORE Models.Client_DF (i.e. before FedMGP's main) is imported, because
Models/Client_DF.py binds the name at import time via
`from torch.utils.data import random_split`. This is the same patching
pattern as tests/test_official_fedta_parity.py's recording wrapper.

The inner patch is deliberately RNG-nonconsuming (no randperm): FedMGP's
torch RNG stream simply continues with whatever state it has; only the
data partitions are pinned.
"""

import json
import sys

_MANIFEST = None
_CTX = {"ctx": None}


def load_manifest(path):
    """Load and validate the split manifest; stores it module-globally."""
    global _MANIFEST
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema") != "fedsmta-split-manifest-v1":
        raise ValueError(f"{path}: unsupported schema {m.get('schema')!r}")
    for key in ("seed", "protocol", "outer", "surrogate", "test_pool",
                "inner"):
        if key not in m:
            raise ValueError(f"{path}: manifest missing {key!r}")
    p = m["protocol"]
    if (p["client_num"], p["task_num"]) != (len(m["outer"]),
                                            len(m["outer"]["0"])):
        raise ValueError(f"{path}: protocol/outer size mismatch")
    _MANIFEST = m
    return m


def manifest():
    if _MANIFEST is None:
        raise RuntimeError("manifest_splitter.load_manifest() not called")
    return _MANIFEST


def patch_torch_random_split():
    """Injection point 3 (inner 70/30). Call BEFORE importing
    Models.Client_DF / FedMGP main."""
    import torch.utils.data as tud

    orig = tud.random_split

    def manifest_random_split(dataset, lengths, *args, **kwargs):
        ctx = _CTX["ctx"]
        # passthrough for any non-(client, task) random_split call
        if ctx is None or _MANIFEST is None:
            return orig(dataset, lengths, *args, **kwargs)
        if len(lengths) != 2:
            return orig(dataset, lengths, *args, **kwargs)
        cid, tid = ctx
        entry = _MANIFEST["inner"][str(cid)][str(tid)]
        n = len(dataset)
        expected = [int(n * 0.7), n - int(n * 0.7)]
        if list(lengths) != expected:
            # not the get_data 70/30 convention -> keep native behavior
            return orig(dataset, lengths, *args, **kwargs)
        return [tud.Subset(dataset, list(entry["train"])),
                tud.Subset(dataset, list(entry["test"]))]

    tud.random_split = manifest_random_split
    return manifest_random_split


def patch_spliter():
    """Injection points 1+2 (outer split, surrogate/test pool). Call with
    the FedMGP repo root on sys.path (imports data.cifar100_subset_spliter
    and utils from FedMGP, keeping its transforms/transforms pipeline
    native)."""
    from data.cifar100_subset_spliter import cifar100_Data_Spliter, \
        CustomedSubset
    from utils import build_transform
    from torchvision.datasets import CIFAR100

    m = manifest()
    p = m["protocol"]

    def random_split_from_manifest(self):
        if (self.client_num, self.task_num, self.private_class_num) != \
                (p["client_num"], p["task_num"], p["private_class_num"]):
            raise ValueError(
                "protocol mismatch: spliter was built with "
                f"({self.client_num}, {self.task_num}, "
                f"{self.private_class_num}) but the manifest froze "
                f"({p['client_num']}, {p['task_num']}, "
                f"{p['private_class_num']})")
        trans = build_transform(True, self.input_size)
        trainset = CIFAR100(root='./local_datasets', train=True,
                            download=True)
        client_subset, client_mask = [], []
        for i in range(p["client_num"]):
            per_subset, per_mask = [], []
            for t in range(p["task_num"]):
                entry = m["outer"][str(i)][str(t)]
                per_subset.append(CustomedSubset(
                    trainset, list(entry["raw_indices"]), trans))
                per_mask.append(list(entry["class_mask"]))
            client_subset.append(per_subset)
            client_mask.append(per_mask)
        return client_subset, client_mask

    def process_testdata_from_manifest(self, surrogate_num):
        if int(surrogate_num) != int(p["surrogate_num"]):
            raise ValueError(
                f"protocol mismatch: surrogate_num={surrogate_num} but the "
                f"manifest froze {p['surrogate_num']}")
        trans = build_transform(False, self.input_size)
        testset = CIFAR100(root='./local_datasets', train=False,
                           download=True)
        surro = CustomedSubset(testset, list(m["surrogate"]["indices"]),
                               trans)
        pool = CustomedSubset(testset, list(m["test_pool"]["indices"]),
                              trans)
        return surro, pool

    cifar100_Data_Spliter.random_split = random_split_from_manifest
    cifar100_Data_Spliter.process_testdata = process_testdata_from_manifest


def patch_client_get_data():
    """Tag each inner random_split with (client_id, task_id) so the
    torch-level patch knows which manifest entry to serve. Call after
    FedMGP's Models.Client_DF is importable."""
    from Models.Client_DF import Client_DF

    orig = Client_DF.get_data

    def ctx_get_data(self, task_id):
        _CTX["ctx"] = (self.id, task_id)
        try:
            return orig(self, task_id)
        finally:
            _CTX["ctx"] = None

    Client_DF.get_data = ctx_get_data


def patch_client_train_update_data():
    """Official-repo bug fix (NOT a protocol change): FedMGP's
    Server_DF.train_clients never calls Client_DF.update_data for
    non-office-home datasets, so self.traindata / self.task_id are never
    set and Client_DF.train crashes at round 0
    (AttributeError: 'Client_DF' object has no attribute 'traindata').

    FedMGP is a FedTA fork; the FedTA Server_DF calls update_data at
    every round start. This patch restores that call by wrapping
    Client_DF.train with FedMGP's OWN update_data (idempotent within a
    task: it only acts at task boundaries). No training logic is
    changed."""
    from Models.Client_DF import Client_DF

    orig = Client_DF.train

    def train_with_update_data(self, round, args):
        self.update_data(round, args)
        return orig(self, round, args)

    Client_DF.train = train_with_update_data


def patch_client_train_nb_classes_bug():
    """Official-repo bug fix #3 (NOT a protocol change): the local-prompt
    phase of Client_DF.train hardcodes `np.setdiff1d(np.arange(200), mask)`
    (an ImageNet-R leftover) while the model head has args.nb_classes=100
    columns for CIFAR-100, so `logits.index_fill` gets out-of-range
    indices 100..199 and triggers a CUDA device-side assert. The global
    phase of the SAME method correctly uses np.arange(args.nb_classes)
    (Models/Client_DF.py L199 vs L250).

    Fix: reload the function's source with exactly that one constant
    replaced by args.nb_classes (asserted to have happened); nothing else
    in the method changes."""
    import inspect
    import sys
    import textwrap
    from Models.Client_DF import Client_DF

    src = textwrap.dedent(inspect.getsource(Client_DF.train))
    bad = "np.setdiff1d(np.arange(200), mask)"
    good = "np.setdiff1d(np.arange(args.nb_classes), mask)"
    if bad not in src:
        raise RuntimeError(
            "FedMGP Client_DF.train no longer contains the hardcoded "
            f"{bad!r} line; re-audit the official repo before patching")
    fixed = src.replace(bad, good)

    mod = sys.modules[Client_DF.__module__]
    exec(compile(fixed, "<Client_DF.train nb_classes bugfix>", "exec"),
         vars(mod))
    Client_DF.train = mod.train


def patch_customed_subset_pil():
    """Official-repo bug fix #2 (NOT a protocol change): FedMGP's
    CustomedSubset.__getitem__ has the PIL conversion commented out
    (`# img = Image.fromarray(img)`), so raw numpy arrays are fed to
    RandomResizedCrop and every training/eval batch crashes
    (TypeError: Unexpected type <class 'numpy.ndarray'>). The FedTA
    lineage original (and the FedSMTA repo) converts to PIL first; this
    restores that exact line, keeping FedMGP's own attribute names."""
    from data.cifar100_subset_spliter import CustomedSubset
    from PIL import Image

    def getitem_pil(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    CustomedSubset.__getitem__ = getitem_pil


def apply_all(manifest_path, fedmgp_root):
    """Convenience: full injection in the correct order for a process that
    has NOT yet imported FedMGP's Models.Client_DF:

      load manifest -> patch torch.utils.data.random_split -> put FedMGP
      root on sys.path + chdir -> import FedMGP main (binds the patched
      random_split at its module-import time) -> patch the spliter class
      -> wrap Client_DF.get_data.

    Returns (manifest, fedmgp_main_module)."""
    import os
    import sys
    from pathlib import Path

    load_manifest(manifest_path)
    patch_torch_random_split()

    root = str(Path(fedmgp_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    import main as fedmgp_main  # noqa: F401  (FedMGP main.py)

    patch_spliter()
    patch_client_get_data()
    return manifest(), fedmgp_main


# ==========================================================================
# ImageNet-R section (manifest dataset == 'ImageNet-R')
#
# FedMGP's official repo never shipped an ImageNet-R route: main.py routes
# only cifar100 / 5datasets / office-home, config/ has no INR entry, and
# data/ has no INR spliter — although Models/Client_DF.py already
# recognizes the internal name 'imagenetr' (class_mask handling + update_
# data), i.e. the route was intended but never finished. What the official
# repo DOES have is a dead-code Imagenet_R class in data/continual_
# datasets.py whose constructor is broken (checks the extracted FOLDER
# with os.path.isfile -> always False -> RuntimeError / re-download) and
# whose ImageFolder-based enumeration is not the one the frozen manifest
# refers to.
#
# Protocol therefore mandates: the manifest indices enumerate the MAIN
# repo's data/continual_datasets.py Imagenet_R (sorted folder/file order,
# numpy data; that is the class the reference run and
# diagnostics/export_split_manifest.py used), and the client subsets use
# the main repo's data/imagenet_r_subset_spliter.py CustomedSubset
# semantics. Both modules are loaded via importlib under alias names so
# the FedMGP checkout itself stays byte-identical (their internal
# `from data... import` lines would otherwise resolve to FedMGP's own
# package inside the chdir'ed run context). `from utils import
# build_transform` inside those modules DOES resolve to FedMGP's utils in
# the run context — which is the method-native choice (FedMGP's own
# augmentation), exactly the policy the cifar100 path already follows.
# ==========================================================================

_MAINREPO_CACHE = {}
_CACHED_INR_CLASS = []


def _load_mainrepo_module(alias, relpath):
    """importlib-load a main-repo module under a private alias name.

    Used only for main-repo modules that must be imported INSIDE the
    FedMGP-chdir'ed process: a plain `from data.x import y` there would
    resolve to FedMGP's own (dead-code / different) module of the same
    name. Modules loaded here must not import other main-repo modules
    through the `data`/`utils` names unless that resolution is intended
    (for utils it IS intended: method-native FedMGP transforms)."""
    import importlib.util
    from pathlib import Path

    if alias in _MAINREPO_CACHE:
        return _MAINREPO_CACHE[alias]
    path = Path(__file__).resolve().parent.parent / relpath
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    _MAINREPO_CACHE[alias] = mod
    return mod


def mainrepo_imagenet_r_class():
    """The main repo's Imagenet_R dataset class (numpy data, sorted
    enumeration, fixed tar-check) — the enumeration the manifest indices
    refer to. data/continual_datasets.py is standalone-importable (no
    main-repo-relative imports), so the alias load is exact."""
    return _load_mainrepo_module(
        "fedsmta_data_continual_datasets",
        "data/continual_datasets.py").Imagenet_R


_INR_INSTANCES = {}


def imagenet_r_instance(train, root="./local_datasets"):
    """Cached constructor over the main repo's Imagenet_R: loading the
    24k/6k images takes minutes, and the adapter route + the gate need
    the SAME deterministic enumeration several times; cache one instance
    per (root, train)."""
    key = (str(root), bool(train))
    if key not in _INR_INSTANCES:
        _INR_INSTANCES[key] = mainrepo_imagenet_r_class()(
            root=root, train=train, download=False)
    return _INR_INSTANCES[key]


def _cached_imagenet_r_class():
    """A subclass of the main repo's Imagenet_R whose construction
    resolves to the shared imagenet_r_instance() cache (one 24k/6k image
    load per (root, train) per process — native regeneration, adapter
    injection and seals all enumerate the identical object)."""
    base = mainrepo_imagenet_r_class()
    if _CACHED_INR_CLASS and _CACHED_INR_CLASS[0].__bases__[0] is base:
        return _CACHED_INR_CLASS[0]

    class _CachedImagenetR(base):
        def __new__(cls, root, train=True, **kwargs):
            return imagenet_r_instance(train, root=root)

        def __init__(self, *args, **kwargs):
            pass

    _CACHED_INR_CLASS.append(_CachedImagenetR)
    return _CachedImagenetR


def load_mainrepo_imagenet_r_spliter_module():
    """Alias-load the main repo's data/imagenet_r_subset_spliter.py and
    rebind its module-level Imagenet_R to the main repo's class (its own
    `from data.continual_datasets import Imagenet_R` resolves to
    FedMGP's dead-code class inside the run context). Returns the module;
    ImagenetR_spliter.random_split / process_testdata are replaced by
    patch_imagenet_r_spliter() afterwards."""
    mod = _load_mainrepo_module(
        "fedsmta_imagenet_r_subset_spliter",
        "data/imagenet_r_subset_spliter.py")
    mod.Imagenet_R = _cached_imagenet_r_class()
    return mod


def patch_imagenet_r_spliter():
    """INR counterpart of patch_spliter(): replace the (main-repo lineage)
    ImagenetR_spliter's random_split / process_testdata with manifest-
    driven versions, so FedMGP's INR route loads exactly the frozen
    client/task/class/sample assignment. Subsets are built with the main
    repo's CustomedSubset semantics over the main repo's Imagenet_R
    enumeration, with FedMGP's own build_transform (method-native, same
    policy as the cifar100 path). Returns the patched class."""
    from utils import build_transform

    mod = load_mainrepo_imagenet_r_spliter_module()
    ImagenetR_spliter = mod.ImagenetR_spliter
    CustomedSubset = mod.CustomedSubset

    m = manifest()
    p = m["protocol"]

    def random_split_from_manifest(self):
        if (self.client_num, self.task_num, self.private_class_num) != \
                (p["client_num"], p["task_num"], p["private_class_num"]):
            raise ValueError(
                "protocol mismatch: spliter was built with "
                f"({self.client_num}, {self.task_num}, "
                f"{self.private_class_num}) but the manifest froze "
                f"({p['client_num']}, {p['task_num']}, "
                f"{p['private_class_num']})")
        trans = build_transform(True, self.input_size)
        trainset = imagenet_r_instance(True, root=getattr(
            self, "data_path", "./local_datasets"))
        client_subset, client_mask = [], []
        for i in range(p["client_num"]):
            per_subset, per_mask = [], []
            for t in range(p["task_num"]):
                entry = m["outer"][str(i)][str(t)]
                per_subset.append(CustomedSubset(
                    trainset, list(entry["raw_indices"]), trans, None))
                per_mask.append(list(entry["class_mask"]))
            client_subset.append(per_subset)
            client_mask.append(per_mask)
        return client_subset, client_mask

    def process_testdata_from_manifest(self, surrogate_num):
        if int(surrogate_num) != int(p["surrogate_num"]):
            raise ValueError(
                f"protocol mismatch: surrogate_num={surrogate_num} but the "
                f"manifest froze {p['surrogate_num']}")
        trans = build_transform(False, self.input_size)
        testset = imagenet_r_instance(False, root=getattr(
            self, "data_path", "./local_datasets"))
        surro = CustomedSubset(testset, list(m["surrogate"]["indices"]),
                               trans, None)
        pool = CustomedSubset(testset, list(m["test_pool"]["indices"]),
                              trans, None)
        return surro, pool

    ImagenetR_spliter.random_split = random_split_from_manifest
    ImagenetR_spliter.process_testdata = process_testdata_from_manifest
    return ImagenetR_spliter


def patch_server_init_client_imagenetr():
    """Official-repo INR blocker (NOT a protocol change): Server_DF.
    init_client builds real clients only for data_name in
    ('cifar100', '5datasets'); 'imagenetr' — which Client_DF.__init__ and
    Client_DF.update_data DO recognize (FedMGP's own INR identifiers) —
    falls into the office-home branch and passes subset=None /
    class_mask=None, so every INR run would crash at the first get_data.
    The official route was clearly intended (Client_DF side is done);
    this patch restores the one missing dataset name in the condition.
    Nothing else in init_client's body changes."""
    import inspect
    import textwrap
    from Models.Server_DF import Server_DF

    src = textwrap.dedent(inspect.getsource(Server_DF.init_client))
    bad = ("if self.args.data_name == 'cifar100' or "
           "self.args.data_name == '5datasets':")
    good = ("if self.args.data_name == 'cifar100' or "
            "self.args.data_name == '5datasets' or "
            "self.args.data_name == 'imagenetr':")
    if bad not in src:
        raise RuntimeError(
            "FedMGP Server_DF.init_client no longer contains the "
            f"cifar/5datasets-only condition {bad!r}; re-audit the "
            "official repo before patching")
    if "imagenetr" in src:
        raise RuntimeError(
            "FedMGP Server_DF.init_client already knows 'imagenetr'; "
            "re-audit the official repo before patching")
    fixed = src.replace(bad, good)

    mod = sys.modules[Server_DF.__module__]
    exec(compile(fixed, "<Server_DF.init_client imagenetr route fix>",
                 "exec"), vars(mod))
    Server_DF.init_client = mod.init_client
    return Server_DF


def load_mainrepo_icifar100c():
    """The main repo's INR-capable iCIFAR100c — the surrogate-data
    wrapper FedMGP's Server_DF.filter / kd_fusion_* drive
    (getTrainData(classes) + (index, x, y) items).

    FedMGP's OWN data/iCIFAR100c.py is CIFAR-only and crashes on INR:
    `self.data[boolean mask]` where self.data is a plain python list of
    ragged numpy images (TypeError). The main repo's copy is the same
    FedTA-lineage class with exactly the INR fixes the reference run
    needed (np.array(...) with dtype=object fallback for ragged images,
    np.array(self.targets) instead of np.ndarray(...),
    Image.fromarray(...).convert('RGB')). Semantics (full-test-set
    unwrap via subset.dataset, per-class boolean selection, train
    transform, (index, img, target) items) are identical to FedMGP's own
    wrapper on cifar100, so the method's surrogate behavior is unchanged.

    Its `from utils import build_transform` resolves to FedMGP's utils
    inside the run context — i.e. FedMGP's own (method-native)
    transform, same policy as the cifar100 path."""
    return _load_mainrepo_module(
        "fedsmta_data_icifar100c", "data/iCIFAR100c.py").iCIFAR100c
