"""Powder / FedPCR (external_baselines/Powder) training entry with the
frozen protocol manifest injected.

Frozen design (external-baseline fairness policy, same as FedMGP/C2Prompt/
FedProTIP/HePCo/GLFC):

  The official Powder implementation is preserved verbatim; ONLY the
  dataset-partition interface is replaced so Powder consumes the exact
  same client/task/class/sample assignment as FedSMTA. No file inside
  external_baselines/Powder is ever edited; fl_main_hard.py is a flat
  SCRIPT (same situation as GLFC's fl_main.py) -- the adapter reads its
  source, applies anchored fail-loud replacements and exec()s it, and the
  two method-level hardcode bugs that the non-uniform protocol triggers
  are patched at runtime on the imported classes (surgical source patch,
  HePCo Trainer.train pattern).

Official command mirrored (src/fl_main_hard.py + option.py):

  method=cprompt_powder_v2 prompt_flag=codap_2d_v2
  prompt_param="25 10 10 8 0 0 6 10 8" num_clients=5 local_clients=5
  epochs_local=10 batch_size=64 lr=0.005 global_update_lr=100
  tasks_global=3 epochs_global=15 (ImageNet-R official)

  The official ImageNet-R config encodes 5 clients x 5 (client,task)
  pairs: epochs_global / tasks_global = 5 tasks, prompt_param[0] = 25 =
  the number of global task ids (num_clients x tasks). The adapter keeps
  that arithmetic with the frozen protocol: tasks_global = R rounds/task
  (default 5), epochs_global = R x T, numclass = K_final (100 / 200),
  class_per_task = task_size = inc[0] (32 / 40). CIFAR-100 has no
  official command (the class-distribution generator has no CIFAR branch
  and would NameError); the ImageNet-R official hyperparameters are used
  verbatim, per the frozen-baseline policy.

Protocol reconciliation for a GLOBAL class-IL model (same core problem
as FedProTIP/HePCo/GLFC, solved with the same FIRST-ARRIVAL scheme):

  * class space = first-arrival order (CIFAR increments [32, 26, 14, 16,
    12]; ImageNet-R [40] x 5);
  * class_distribution_client[c][t] = the client's manifest mask(c, t)
    remapped to arrival positions; class_distribution_client_real[c][t]
    = the raw classes in the same order; class_distribution_client_
    proportion[c][t] = [0, 1] (the client's full owned slice of each
    class, i.e. the official proportion semantics at [0, 1]);
  * the official partition generators (centralized / dil / ImageNet-R
    range slicing / DomainNet) are sealed off behind the same hook;
  * the model's full-width fc (numclass columns, built once) is trained
    with CE over the CURRENT task's masked logits + per-round ntd
    distillation to the old global model, so a rotated public class
    whose column already exists is LEGALLY re-trained by the rotating
    client (uses per-round knowledge distillation; no drop). Paper note
    for Powder rows: "uses per-round old-model distillation; rotated
    public classes are legally re-trained".

Starvation edge (CIFAR, exactly the two GLFC cases): mask classes with
ZERO owned inner.train samples ((c0,t4,raw10), (c2,t3,raw56)) cannot be
trained and would crash calculate_prompt_choosing (torch.mean(None)).
They are omitted from that (c, t) structure entry, recorded in
drops.json (zero_train_sample_classes_omitted); their global columns
are learned at arrival by the owning client (raw10 at t0 by c1, raw56
at t0 by c1) so boundary evaluation is unaffected.

ImageNet-R bridge: no official CIFAR-style loader is needed -- the
manifest duck-type pre-decodes local_datasets/imagenet-r (the
sorted-walk domain == the manifest raw domain, same walk as
run_fedprotip/run_glfc) into 84x84 uint8 arrays (the GLFC-bridge
convention) and feeds them through Powder's own transforms
(RandomResizedCrop(224) train / Resize(256)+CenterCrop(224) test), which
resize to the ViT-B/16 224 input. CIFAR-100 feeds its native 32x32
arrays through the same transform chain. The ViT-B/16 augreg checkpoint
(torch hub cache) is loaded by the official code via timm.

Injection points (all fail-loud, seal-checked by gate_powder.py):

  fl_main_hard.py source patches (anchored, count-checked):
    P1 partition hook      -> _POWDER_MANIFEST_STRUCTURES(args) defines
                              class_distribution_client/_real/_proportion
                              (dict-of-lists, starved-filtered) + di=None,
                              inserted before the FULL TEST print; the
                              official generators stay in the file, dead
    P2 dataset branch      -> manifest duck-type datasets
                              (ManifestPowderDataset train/test pair; the
                              official iCIFAR100/iIMAGENET_R loaders --
                              which would download and use the global
                              datasets -- are sealed off)
    P3 client churn        -> old_client_1 = all clients (frozen cohort)
    P4 participation       -> clients_index = all clients every round
                              (frozen frac=1.0)
    P5 training log        -> {run_dir}/training_log (official tree
                              layout preserved inside it)
    P6 global eval call    -> _POWDER_GLOBAL_EVAL (official
                              model_global_eval_hard verbatim on the
                              manifest test pool + a once-per-boundary
                              RNG-neutral per-(c,k<=t) manifest boundary
                              eval on the global model, mirroring
                              GLFC's _glfc_boundary_eval: shuffle=False,
                              model_g kept at client_index=-1 so the
                              forward masks unarrived columns with -inf)

  Sealed method-level hardcode bugfixes (runtime source patches on the
  imported classes; numerically identical on every official config):
    S1 CodaPrompt_2d_v2.__init__ (models_Cprompt/vit_coda_p.py): the
       task-similarity weight / weight_c / task_embedding are sized
       int(numclass/class_per_task) -- a uniform-task hardcode. The
       frozen protocol has non-uniform increments (100/32 = 3 rows for
       25 global task ids) and the prompt forward indexes them with
       global_task_id = task_id*num_clients + client_index (up to 24),
       which would IndexError from round 1. Patched to e_task_number
       (= prompt_param[0] = 25 = num_clients x tasks_global, exactly
       what the official ImageNet-R config made numclass/class_per_task
       equal). Gate PartC asserts the three official anchors.
    S2 DualPrompt_hard._ntd_loss (prompt_hard.py): the else branch
       (cprompt_powder_v2's path) restricts distillation columns via
       refine_as_not_true(numclass) + a [first:last] SLICE of the
       column-removed logits -- arithmetic that equals "current classes
       minus the true class" only when current_class is a contiguous
       ascending range (the official configs). The manifest's scattered
       arrival positions would silently distill over wrong columns.
       Patched to the official authors' own commented-out
       refine_as_not_true2(logits, targets, current_class) (direct
       indexing, identical on contiguous official ranges).

  Module stubs: nni (the script is NNI-aware; get_next_parameter -> {})
  and cv2 (tiny_imagenet/mini_imagenet/proxy_data/fractal_learning
  import it; the fedspace fractal chain needs cvtColor/COLOR_HSV2RGB/
  GaussianBlur/resize/INTER_LINEAR as importable names) and
  fgvcdata/pytorch_lightning (fedspace_hard -> fractal_learning ->
  classify_datamodule) -- all fail-loud on use, none on the Powder
  execution path.

Protocol mapping (frozen, enforced against the manifest):
  5 clients x 5 tasks, full participation every round, seed = manifest
  seed (42), 5 FL rounds per task (tasks_global; official 3), 10 local
  epochs (official), Adam lr=0.005 on [prompt + fc] (ViT backbone frozen
  per the official optimizer construction), server aggregation =
  FedAvg_our_v1 with global_update_lr=100 (official), eval pool =
  manifest test_pool (surrogate-excluded).
Method-native (official option.py defaults / slurm, overridable):
  prompt_param 25 10 10 8 0 0 6 10 8, batch_size=64, top_k=10,
  class-incremental prompt selection via aqk, ntd distillation tau=3.

Usage (from the FedSMTA repo root, FedSMTA conda env; GPU runs must be
executed OUTSIDE the sandbox -- CUDA init is blocked inside it):

  python protocol_adapters/run_powder.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      [--data-root local_datasets] [--device 0] [--rounds 5] \
      [--epochs-local 10] [--batch-size 64] [--lr 0.005] \
      [--global-update-lr 100] [--limit-tasks N]   # SMOKE ONLY

Output: output/{cifar100|ImageNet-R}/powder/{run_label}/seed_{seed}/
  (metrics/ CSV tree + meta.json + drops.json + training_log/)
"""

import argparse
import inspect
import json
import os
import sys
import textwrap
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import metrics_recorder  # noqa: E402

POWDER_ROOT = REPO_ROOT / "external_baselines" / "Powder" / "src"

_STATE = {
    "manifest": None,
    "increments": None,      # new-class counts per task (arrival order)
    "cumulative": None,      # cumulative head width K_t after task t
    "arrival": None,         # raw class -> arrival position
    "pos_to_raw": None,      # arrival position -> raw class
    "train_raw": None,       # (c, t) -> inner.train raw trainset indices
    "own_by_class": None,    # client -> raw class -> [row views into base]
    "pool_by_class": None,   # raw class -> [pool positions]
    "datasets": None,        # {'train': ..., 'test': ...}
    "recorder": None,
    "run_dir": None,
    "training_log_dir": None,
    "nb_classes": None,
    "n_tasks": None,         # effective task count (truncated on smoke)
    "base": None,            # raw arrays/targets of the train + test sets
    "starved": None,         # (c, t) -> set of raw classes with 0 own samples
    "drops": None,           # drops.json payload
    "ns": None,              # fl_main_hard exec namespace
    "boundary_done": None,    # ep_g values whose boundary eval already ran
    "svd_rerouted": False,
    "device": 0,
}

# ---------------------------------------------------------------------------
# sealed env bugfix: numpy LAPACK self-test + scipy reroute (see
# run_fedprotip.install_robust_svd; Powder itself barely touches LAPACK on
# the hot path but the uniform env guard is cheap and catches a broken host)
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
            diag.append(f"{shape}:np={float(s.max()):.3f} "
                        f"torch={float(ref.max()):.3f}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            diag.append(f"{shape}:EXC {type(exc).__name__}")
        healthy &= ok
    if healthy:
        print(f"[powder-adapter] numpy LAPACK self-test OK ({'; '.join(diag)})")
        return
    print("[powder-adapter] numpy LAPACK BROKEN (%s) -- rerouting "
          "np.linalg.svd -> scipy.linalg.svd (sealed, see "
          "run_fedprotip.install_robust_svd)" % "; ".join(diag))
    import scipy.linalg as sla

    def _scipy_svd(a, full_matrices=True, compute_uv=True, *a_, **k):
        return sla.svd(a, full_matrices=full_matrices, compute_uv=compute_uv,
                       lapack_driver="gesdd")

    np.linalg.svd = _scipy_svd
    _STATE["svd_rerouted"] = True


# ---------------------------------------------------------------------------
# module stubs (fail-loud; none of these is on the Powder execution path)
# ---------------------------------------------------------------------------


def install_import_stubs():
    os.environ.setdefault("MPLBACKEND", "Agg")

    if "nni" not in sys.modules:
        nni = types.ModuleType("nni")

        def _get_next_parameter():
            return {}

        def _report_intermediate_result(payload):
            print(f"[nni-stub] intermediate: {payload}")

        def _report_final_result(payload):
            print(f"[nni-stub] final: {payload}")

        nni.get_next_parameter = _get_next_parameter
        nni.report_intermediate_result = _report_intermediate_result
        nni.report_final_result = _report_final_result
        sys.modules["nni"] = nni

    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")

        def _cv2_missing(*args, **kwargs):
            raise RuntimeError(
                "[powder-adapter] cv2 is not installed in the frozen env "
                "and is never needed on the cprompt_powder_v2 path -- a "
                "real cv2 call happened; refusing to fake it")

        for name in ("cvtColor", "GaussianBlur", "resize"):
            setattr(cv2, name, _cv2_missing)
        cv2.COLOR_HSV2RGB = 4
        cv2.INTER_LINEAR = 2
        sys.modules["cv2"] = cv2

    # fedspace_hard -> fractal_learning.training.datamodule ->
    # classify_datamodule reads fgvcdata.CUB / .StanfordDogs /
    # .StanfordCars / .Aircraft at class-body import time (assigned as
    # plain attributes) and subclasses pytorch_lightning.
    # LightningDataModule; neither module is needed at runtime.
    if "fgvcdata" not in sys.modules:
        fgv = types.ModuleType("fgvcdata")
        for ds_name in ("CUB", "StanfordDogs", "StanfordCars", "Aircraft"):
            setattr(fgv, ds_name,
                    type(f"_fgvcdata_stub_{ds_name}", (), {}))
        fgv.__dict__["__getattr__"] = \
            lambda name: (_ for _ in ()).throw(RuntimeError(
                f"[powder-adapter] fgvcdata.{name} is not installed and "
                "is never needed on the cprompt_powder_v2 path -- a real "
                "fgvcdata usage happened; refusing to fake it"))
        sys.modules["fgvcdata"] = fgv

    if "pytorch_lightning" not in sys.modules:
        pl = types.ModuleType("pytorch_lightning")

        class _LDModule:  # noqa: N801
            pass

        pl.LightningDataModule = _LDModule

        def _pl_missing(name):
            # any further pl.* name is only referenced by dead fedspace
            # code paths; return an inert class so isinstance() checks
            # (always False) keep working without installing lightning
            return type(f"_pl_stub_{name}", (), {})

        pl.__dict__["__getattr__"] = _pl_missing
        sys.modules["pytorch_lightning"] = pl

    # fractal_learning.fractals.{ifs,diamondsquare} are @numba.njit-
    # decorated at import time; the stub keeps the pure-numpy functions
    # as plain python (they are only called on the dead fedspace path)
    if "numba" not in sys.modules:
        nba = types.ModuleType("numba")

        def _njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def deco(fn):
                return fn
            return deco

        nba.njit = _njit
        sys.modules["numba"] = nba


# ---------------------------------------------------------------------------
# manifest plumbing (identical semantics to run_glfc.py)
# ---------------------------------------------------------------------------


def load_manifest(path):
    m = json.loads(Path(path).read_text("utf-8"))
    _STATE["manifest"] = m
    return m


def compute_arrival():
    """First-arrival scheme: global class positions ordered by the task at
    which each raw class first enters any client's mask, raw id ascending
    as the within-task tie-break (the convention every adapter in this
    repo uses). Returns (increments, arrival, pos_to_raw)."""
    m = _STATE["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    first_task = {}
    for t in range(nt):
        for c in range(nc):
            for cl in m["outer"][str(c)][str(t)]["class_mask"]:
                cl = int(cl)
                if cl not in first_task:
                    first_task[cl] = t
    order = sorted(first_task, key=lambda r: (first_task[r], r))
    arrival = {r: p for p, r in enumerate(order)}
    pos_to_raw = {p: r for r, p in arrival.items()}
    inc = [sum(1 for r in order if first_task[r] == t) for t in range(nt)]
    cumulative = [int(sum(inc[:t + 1])) for t in range(nt)]
    assert len(order) == m["protocol"]["nb_classes"], \
        "manifest masks do not cover nb_classes"
    assert sum(inc) == m["protocol"]["nb_classes"]
    _STATE["increments"] = inc
    _STATE["cumulative"] = cumulative
    _STATE["arrival"] = arrival
    _STATE["pos_to_raw"] = pos_to_raw
    return inc, arrival, pos_to_raw


def build_train_raw():
    """(c, t) -> the raw trainset indices of the client's inner.train
    samples (inner.train positions index the (c,t) outer raw_indices)."""
    m = _STATE["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    train_raw = {}
    for c in range(nc):
        for t in range(nt):
            outer_raw = m["outer"][str(c)][str(t)]["raw_indices"]
            train_raw[(c, t)] = [int(outer_raw[int(i)])
                                 for i in m["inner"][str(c)][str(t)]["train"]]
    _STATE["train_raw"] = train_raw
    return train_raw


def load_base(data_root):
    """Raw arrays/targets of the full train set and the eval test set.

    CIFAR-100: native 32x32 uint8 arrays (the official transform chain
    RandomResizedCrop(224) upsamples). ImageNet-R: the sorted-walk domain
    over local_datasets/imagenet-r/{train,test} (identical to
    run_fedprotip.walk_imagenet_r / run_glfc.load_base), pre-decoded to
    84x84 arrays (the cross-adapter INR bridge); Powder's own transforms
    resize to 224.
    """
    import torch  # noqa: F401  (sanity: torch importable)
    from PIL import Image

    m = _STATE["manifest"]
    ds = m["dataset"]
    root = os.path.expanduser(data_root)
    base = {"train_x": None, "train_y": None, "test_x": None, "test_y": None}
    if ds == "cifar100":
        from torchvision.datasets import CIFAR100
        tr = CIFAR100(root=root, train=True, download=False)
        te = CIFAR100(root=root, train=False, download=False)
        base["train_x"] = np.asarray(tr.data)
        base["train_y"] = np.asarray(tr.targets, dtype=np.int64)
        base["test_x"] = np.asarray(te.data)
        base["test_y"] = np.asarray(te.targets, dtype=np.int64)
    else:
        def _walk(split):
            files = []
            ys = []
            sroot = os.path.join(root, "imagenet-r", split)
            folders = sorted(d for d in os.listdir(sroot)
                             if os.path.isdir(os.path.join(sroot, d)))
            for ri, fold in enumerate(folders):
                froot = os.path.join(sroot, fold)
                for name in sorted(os.listdir(froot)):
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        files.append(os.path.join(froot, name))
                        ys.append(ri)
            return files, np.asarray(ys, dtype=np.int64)

        def _dec(path):
            with Image.open(path) as im:
                return np.asarray(
                    im.convert("RGB").resize((84, 84), Image.BILINEAR),
                    dtype=np.uint8)

        tfiles, ty = _walk("train")
        efiles, ey = _walk("test")
        base["train_x"] = np.stack([_dec(f) for f in tfiles])
        base["train_y"] = ty
        base["test_x"] = np.stack([_dec(f) for f in efiles])
        base["test_y"] = ey
        print(f"[powder-adapter] pre-decoded ImageNet-R: "
              f"train {base['train_x'].shape}, test {base['test_x'].shape}")
    _STATE["base"] = base
    return base


def build_datasets():
    """Own-samples-per-class (union over the client's tasks, the official
    persistent-slice analog, GLFC convention), the test pool, the
    starvation bookkeeping and the duck-type dataset pair with Powder's
    own transforms."""
    m = _STATE["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    base = _STATE["base"]
    train_x, train_y = base["train_x"], base["train_y"]
    train_raw = _STATE["train_raw"]

    own_by_class = {c: {} for c in range(nc)}
    for c in range(nc):
        for t in range(nt):
            for idx in train_raw[(c, t)]:
                own_by_class[c].setdefault(int(train_y[idx]),
                                           []).append(train_x[idx])
    _STATE["own_by_class"] = own_by_class

    starved = {}
    drops = []
    arrival = _STATE["arrival"]
    for c in range(nc):
        for t in range(nt):
            for cl in m["outer"][str(c)][str(t)]["class_mask"]:
                cl = int(cl)
                if len(own_by_class[c].get(cl, ())) == 0:
                    starved.setdefault((c, t), set()).add(cl)
                    drops.append({
                        "client": c, "task": t, "raw_class": cl,
                        "arrival_position": arrival[cl],
                        "reason": "zero owned inner.train samples at this "
                                  "(client, task); column learned at "
                                  "arrival by the owning client",
                    })
    _STATE["starved"] = {k: sorted(v) for k, v in starved.items()}
    _STATE["drops"] = {
        "total": len(drops),
        "zero_train_sample_classes_omitted": drops,
        "rotated_public_class_revisit_policy":
            "legal continued training (full-width CE over current classes "
            "+ per-round ntd distillation to the old global model); no drop",
    }

    test_pool = m["test_pool"]
    pool_pos = [int(i) for i in test_pool["indices"]]
    if max(pool_pos) >= len(base["test_y"]):
        raise RuntimeError("test_pool exceeds the official test set")
    pool_by_class = {}
    for p, pos in enumerate(pool_pos):
        pool_by_class.setdefault(int(base["test_y"][pos]), []).append(p)
    _STATE["pool_pos"] = pool_pos
    _STATE["pool_by_class"] = pool_by_class

    # Powder's own transforms, constructed exactly like fl_main does
    sys.path.insert(0, str(POWDER_ROOT))
    import dataloaders.utils as plu
    ds_name = "CIFAR100" if m["dataset"] == "cifar100" else "ImageNet_R"
    train_tf = plu.get_transform(dataset=ds_name, phase="train", aug=True,
                                 resize_imnet=True)
    test_tf = plu.get_transform(dataset=ds_name, phase="test", aug=False,
                                resize_imnet=True)

    train_ds = ManifestPowderDataset("train", transform=train_tf)
    test_ds = ManifestPowderDataset("test", transform=test_tf,
                                    test_transform=test_tf)
    _STATE["datasets"] = {"train": train_ds, "test": test_ds}
    return _STATE["datasets"]


# ---------------------------------------------------------------------------
# manifest duck-type dataset (the official iDataset-hard interface:
# positions for labels, raw ids for data selection, client_index dispatch)
# ---------------------------------------------------------------------------


class ManifestPowderDataset:
    """Duck-type of the official dataset objects fl_main passes around
    (iCIFAR100 / iIMAGENET_R / iDOMAINNET hard variants). ONE train-side
    instance is shared by all 5 DualPrompt_hard trainers (the official
    architecture: a single global dataset object with per-client dispatch
    through the client_index argument of getTrainData).

    Label space = arrival positions; data selection = raw class ids;
    per-class data = the client's OWN inner.train samples (union over its
    tasks, the manifest analog of the official persistent Dirichlet
    slice), with the official [p0, p1] proportion slicing semantics
    ([0, 1] -> the full owned slice).
    """

    def __init__(self, split, transform=None, test_transform=None):
        self.split = split
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.transform = transform
        self.test_transform = test_transform

    # -- train side ------------------------------------------------------
    def getTrainData(self, classes, exemplar_set, exemplar_label_set,
                     client_index, classes_real=None,
                     classes_proportion=None,
                     class_distribution_client_di=None, exe_class=None):
        if class_distribution_client_di is not None:
            raise RuntimeError(
                "[powder-adapter] the 'di' partition path is an official "
                "variant the frozen protocol never exercises "
                "(di is always None); refusing to fake it")
        if classes_real is None:
            raise RuntimeError(
                "[powder-adapter] the raw-id getTrainData branch is the "
                "centralized variant; the manifest protocol always passes "
                "classes_real; refusing to fake it")
        own = _STATE["own_by_class"][client_index]
        self.TrainData = []
        self.TrainLabels = []
        for i, raw in enumerate(classes_real):
            data = own.get(int(raw), [])
            if classes_proportion is not None:
                p0, p1 = classes_proportion[0], classes_proportion[1]
                data = data[int(p0 * len(data)):int(p1 * len(data))]
            if exemplar_set:
                if int(raw) not in (exe_class or []):
                    self.TrainData.extend(exemplar_set)
                    self.TrainLabels.extend(
                        np.asarray(exemplar_label_set[i]).tolist()
                        if i < len(exemplar_label_set) else [])
            self.TrainData.extend(data)
            self.TrainLabels.extend([classes[i]] * len(data))

    # -- eval side ---------------------------------------------------------
    def getTestData_hard(self, classes, classes_real):
        base = _STATE["base"]
        pool_by_class = _STATE["pool_by_class"]
        pool_pos = _STATE["pool_pos"]
        self.TestData = []
        self.TestLabels = []
        for i, raw in enumerate(classes_real):
            for p in pool_by_class.get(int(raw), []):
                self.TestData.append(base["test_x"][pool_pos[p]])
                self.TestLabels.append(classes[i])

    # -- exemplar/consolidation interfaces (dead on the frozen path;
    #    implemented for duck-type completeness) --------------------------
    def get_image_class(self, label, client_index, classes_proportion=None):
        data = _STATE["own_by_class"][client_index].get(int(label), [])
        if classes_proportion is not None:
            p0, p1 = classes_proportion[0], classes_proportion[1]
            data = data[int(p0 * len(data)):int(p1 * len(data))]
        if not data:
            return np.empty((0,) + _STATE["base"]["train_x"].shape[1:],
                            dtype=np.uint8)
        return np.stack(data)

    def get_image_class_dict(self, label, client_index,
                             classes_proportion=None, exemplar_set=None):
        data = self.get_image_class(label, client_index, classes_proportion)
        datas = [i for i in data]
        if exemplar_set is not None:
            datas = datas + exemplar_set
        return datas

    def getDistillTrainData(self, classes, exemplar_set, exe_class,
                            client_index, classes_real=None,
                            class_distribution_client_di=None):
        if exemplar_set:
            raise RuntimeError(
                "[powder-adapter] the consolidation/exemplar path belongs "
                "to extension/extencl methods and is never reached by "
                "cprompt_powder_v2; refusing to fake it")
        self.getTrainData(classes, [], [], client_index,
                          classes_real=classes_real,
                          classes_proportion=None,
                          class_distribution_client_di=None, exe_class=None)

    def getDistillTrainDataLabel(self, classes, exemplar_set, exe_class,
                                 client_index, classes_real=None,
                                 class_distribution_client_di=None):
        self.getDistillTrainData(classes, exemplar_set, exe_class,
                                client_index, classes_real=classes_real,
                                class_distribution_client_di=None)
        return self.TrainLabels

    # -- torch.utils.data.Dataset ------------------------------------------
    def getTrainItem(self, index):
        from PIL import Image
        img = Image.fromarray(np.asarray(self.TrainData[index]))
        if self.transform is not None:
            img = self.transform(img)
        return index, img, self.TrainLabels[index]

    def getTestItem(self, index):
        from PIL import Image
        img = Image.fromarray(np.asarray(self.TestData[index]))
        tf = self.test_transform if self.test_transform is not None \
            else self.transform
        if tf is not None:
            img = tf(img)
        return index, img, self.TestLabels[index]

    def __getitem__(self, index):
        if len(self.TrainData) > index and len(self.TrainData) > 0:
            return self.getTrainItem(index)
        return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        return len(self.TestData)


# ---------------------------------------------------------------------------
# the P1 hook: manifest-derived class_distribution structures, injected
# into the fl_main_hard exec namespace
# ---------------------------------------------------------------------------


def _powder_manifest_structures(args):
    ns = _STATE["ns"]
    m = _STATE["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    arrival = _STATE["arrival"]
    starved = _STATE["starved"]
    cdc, cdc_real, cdc_prop = {}, {}, {}
    for c in range(nc):
        for t in range(nt):
            raws = sorted(int(x) for x in m["outer"][str(c)][str(t)]
                          ["class_mask"])
            omit = starved.get((c, t), ())
            keep = [r for r in raws if r not in omit]
            cdc.setdefault(c, []).append([arrival[r] for r in keep])
            cdc_real.setdefault(c, []).append(keep)
            cdc_prop.setdefault(c, []).append([0, 1])
    ns["class_distribution_client"] = cdc
    ns["class_distribution_client_real"] = cdc_real
    ns["class_distribution_client_proportion"] = cdc_prop
    ns["class_distribution_client_di"] = None
    print(f"[powder-adapter] manifest partition injected: "
          f"{nc} clients x {nt} tasks, positions "
          f"inc={_STATE['increments']}, starved omissions="
          f"{sum(len(v) for v in starved.values())}")


# ---------------------------------------------------------------------------
# P6 wrapper: official per-(client,task) eval verbatim + once-per-boundary
# RNG-neutral manifest boundary eval on the global model
# ---------------------------------------------------------------------------


def _powder_global_eval(model_for_eval, test_dataset, model_task_id,
                        task_size, device, method, task_num, current_class,
                        current_class_real, model_g, ep_g, tasks_global):
    from Fed_utils import model_global_eval_hard
    acc_global, accs_global = model_global_eval_hard(
        model_for_eval, test_dataset, model_task_id, task_size, device,
        method, task_num, current_class, current_class_real)
    if (ep_g + 1) % tasks_global == 0 \
            and ep_g not in _STATE["boundary_done"]:
        _STATE["boundary_done"].add(ep_g)
        _powder_boundary_eval(model_g, test_dataset,
                              (ep_g + 1) // tasks_global - 1, device)
    return acc_global, accs_global


def _powder_boundary_eval(model_g, test_dataset, task_id, device):
    """Per-(c, k <= t) boundary eval on the manifest test pool, mirroring
    GLFC's _glfc_boundary_eval: the global model at client_index == -1
    (the forward then masks unarrived columns with -inf), shuffle=False,
    no_grad, mode saved/restored -- RNG-neutral. Labels are arrival
    positions, which are all < K_t so argmax over the masked full-width
    logits is exactly the truncated-arrival eval."""
    import torch
    from torch.utils.data import DataLoader
    m = _STATE["manifest"]
    nc = m["protocol"]["client_num"]
    arrival = _STATE["arrival"]
    rec = _STATE["recorder"]
    was_training = model_g.training
    model_g.eval()
    n_evaluated = 0
    for c in range(nc):
        for k in range(task_id + 1):
            raws = sorted(int(x) for x in m["outer"][str(c)][str(k)]
                          ["class_mask"])
            positions = [arrival[r] for r in raws]
            test_dataset.getTestData_hard(positions, raws)
            if len(test_dataset.TestLabels) == 0:
                raise RuntimeError(
                    f"[powder-adapter] empty boundary subset for "
                    f"(client {c}, task {k}) -- manifest test pool is "
                    f"missing classes {raws}")
            loader = DataLoader(test_dataset, batch_size=256,
                                 shuffle=False, num_workers=0)
            correct = 0
            total = 0
            with torch.no_grad():
                for _, imgs, labels in loader:
                    if isinstance(device, int):
                        imgs = imgs.cuda(device)
                    else:
                        imgs = imgs.cuda()
                    out = model_g(imgs, device=device)
                    pred = out.argmax(dim=1).cpu()
                    lab = labels if torch.is_tensor(labels) \
                        else torch.tensor(labels)
                    correct += int((pred == lab).sum().item())
                    total += int(lab.numel())
            acc = correct / total if total else 0.0
            rec.record(c, task_id, k, acc, phase="boundary")
            n_evaluated += 1
    model_g.train(was_training)
    rec.flush(task_id + 1)
    print(f"[powder-adapter] boundary eval after task {task_id}: "
          f"{n_evaluated} (client, test-task) records flushed")


# ---------------------------------------------------------------------------
# sealed method-level bugfixes (runtime source patches on imported classes)
# ---------------------------------------------------------------------------


def _patch_method(cls, method_name, replaces, note):
    """Read the official method source, apply count-checked anchored
    replacements, exec it against the module's globals and rebind it on
    the class. The official file is never touched; a drifted anchor
    fails loudly (gate_powder.py PartC asserts the anchors too)."""
    src = inspect.getsource(getattr(cls, method_name))
    # zero-arg super() relies on a __class__ closure cell that exec'ed
    # code does not have; rewrite to the explicit 2-arg form
    # (numerically identical, applies to every patched method)
    if src.count("super()"):
        src = src.replace("super()", f"super({cls.__name__}, self)")
    for old, new in replaces:
        if src.count(old) != 1:
            raise RuntimeError(
                f"[powder-adapter] {note}: anchor not found exactly once "
                f"in {cls.__name__}.{method_name}:\n{old!r}\n"
                f"(count={src.count(old)}); official repo drifted?")
        src = src.replace(old, new)
    mod = sys.modules[cls.__module__]
    code = textwrap.dedent(src)
    exec(compile(code, f"<powder-adapter:{cls.__name__}.{method_name}>",
                 "exec"), vars(mod))
    setattr(cls, method_name, vars(mod)[method_name])


def patch_codaprompt_init():
    """S1: task-similarity weight matrices sized int(numclass/class_per_task)
    (a uniform-task hardcode; 100/32=3 rows for 25 global task ids ->
    IndexError from round 1 under the non-uniform protocol) ->
    e_task_number (= prompt_param[0] = num_clients x tasks_global, which
    every official config made numclass/class_per_task equal)."""
    from models_Cprompt.vit_coda_p import CodaPrompt_2d_v2
    anchor = ("torch.eye(int(self.args.numclass/self.args.class_per_task), "
              "device=self.device)")
    _patch_method(
        CodaPrompt_2d_v2, "__init__",
        [(f"self.weight = {anchor}",
          "self.weight = torch.eye(self.e_task_number, device=self.device)"
          "  # [protocol-adapter] sealed S1: rows must cover num_clients x"
          " tasks_global global task ids; official configs sized"
          " numclass/class_per_task to exactly this"),
         (f"self.weight_c = {anchor}",
          "self.weight_c = torch.eye(self.e_task_number,"
          " device=self.device)  # [protocol-adapter] sealed S1"),
         ("self.task_embedding = torch.FloatTensor(int(self.args.numclass/"
           "self.args.class_per_task), emb_d).to(self.device)",
          "self.task_embedding = torch.FloatTensor(self.e_task_number,"
          " emb_d).to(self.device)  # [protocol-adapter] sealed S1")],
        "S1 CodaPrompt_2d_v2 weight sizing")


def patch_ntd_loss():
    """S2: the else branch of DualPrompt_hard._ntd_loss restricts the
    distillation columns via refine_as_not_true(numclass) + a
    [first:last] slice of the column-removed logits -- equal to 'current
    classes minus the true class' only for CONTIGUOUS ascending
    current_class (official configs). Scattered arrival positions would
    silently distill over wrong columns. Switched to the official
    authors' own commented-out refine_as_not_true2(logits, targets,
    current_class) (direct indexing; identical on contiguous ranges)."""
    from prompt_hard import DualPrompt_hard
    _patch_method(
        DualPrompt_hard, "_ntd_loss",
        [("logits = self.refine_as_not_true(logits, targets, self.numclass)",
          "logits = self.refine_as_not_true2(logits, targets,"
          " self.current_class)  # [protocol-adapter] sealed S2:"
          " scattered arrival positions"),
         ("logits = logits[:, self.current_class[0]:"
          "self.current_class[len(self.current_class) - 1]]",
          "pass  # [protocol-adapter] sealed S2: refine_as_not_true2"
          " already restricts to current_class"),
         ("dg_logits = self.refine_as_not_true(dg_logits, targets,"
          " self.numclass)",
          "dg_logits = self.refine_as_not_true2(dg_logits, targets,"
          " self.current_class)  # [protocol-adapter] sealed S2:"
          " scattered arrival positions"),
         ("dg_logits = dg_logits[:, self.current_class[0]:"
          "self.current_class[len(self.current_class) - 1]]",
          "pass  # [protocol-adapter] sealed S2: refine_as_not_true2"
          " already restricts to current_class")],
        "S2 DualPrompt_hard._ntd_loss column restriction")


# ---------------------------------------------------------------------------
# fl_main_hard.py source patches (P1..P6, count-checked)
# ---------------------------------------------------------------------------


def patch_fl_main_source(src):
    def rep(old, new, count):
        found = src.count(old)
        if found != count:
            raise RuntimeError(
                f"[powder-adapter] fl_main_hard.py anchor not found "
                f"{count}x (found {found}x):\n{old!r}\n"
                "official repo drifted?")
        return src.replace(old, new)

    # P1: manifest partition hook (official generators stay, dead)
    src = rep('print("********* FULL TEST **********")',
              "_POWDER_MANIFEST_STRUCTURES(args)\n"
              'print("********* FULL TEST **********")', 1)

    # P2: dataset branch -> manifest duck-type datasets
    src = rep(
        "if args.dataset == 'CIFAR100':\n"
        "    train_dataset = iCIFAR100('dataset', transform=train_transform,"
        " download=True)\n"
        "    test_dataset = iCIFAR100('dataset', test_transform="
        "test_transform, train=False, download=True)",
        "if args.dataset == 'CIFAR100' or args.dataset == 'ImageNet_R':\n"
        "    # [protocol-adapter] P2: official dataset loaders sealed off;"
        " the\n"
        "    # manifest duck-type pair is injected (arrival-space labels,"
        " test\n"
        "    # pool eval, per-client dispatch through client_index)\n"
        "    train_dataset = _POWDER_STATE['datasets']['train']\n"
        "    test_dataset = _POWDER_STATE['datasets']['test']", 1)

    # P3: client churn sealed off (frozen 5-client cohort)
    src = rep(
        "old_client_1 = random.sample([i for i in range(overall_client)],"
        " int(overall_client * 0.5))",
        "old_client_1 = [i for i in range(overall_client)]"
        "  # [protocol-adapter] P3: churn sealed off (frozen cohort)",
        1)

    # P4: full participation every round (frozen frac=1.0)
    src = rep(
        "clients_index = random.sample(range(num_clients),"
        " args.local_clients)",
        "clients_index = list(range(num_clients))"
        "  # [protocol-adapter] P4: full participation (frozen frac=1.0)",
        1)

    # P5: training log tree preserved inside the run dir
    src = rep(
        "output_dir = osp.join('./training_log', args.method, 'seed' + "
        "str(args.seed))",
        "output_dir = _POWDER_STATE['training_log_dir']"
        "  # [protocol-adapter] P5: official log tree inside the run dir",
        1)

    # P6: global eval -> wrapper (official eval + boundary records).
    # The identical call text exists twice (the centralized branch is
    # dead for cprompt_powder_v2); both are wrapped.
    call = ("acc_global, accs_global = model_global_eval_hard("
            "model_for_eval, test_dataset, model_task_id, args.task_size, "
            "args.device, args.method, int(args.epochs_global / "
            "args.tasks_global), current_class, current_class_real)")
    wrapped = ("acc_global, accs_global = _POWDER_GLOBAL_EVAL("
               "model_for_eval, test_dataset, model_task_id, "
               "args.task_size, args.device, args.method, "
               "int(args.epochs_global / args.tasks_global), "
               "current_class, current_class_real, model_g, ep_g, "
               "args.tasks_global)")
    src = rep(call, wrapped, 2)
    return src


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        "Powder/FedPCR training entry with the frozen protocol manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("--device", type=int, default=0,
                   help="CUDA device id (official --device is nargs='+')")
    p.add_argument("--rounds", type=int, default=5,
                   help="FL rounds per task (= official tasks_global)")
    p.add_argument("--epochs-local", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--global-update-lr", type=float, default=100)
    p.add_argument("--limit-tasks", type=int, default=0,
                   help="0 = all manifest tasks; SMOKE ONLY otherwise")
    p.add_argument("--run-label", default="manifest_adapter")
    args = p.parse_args()
    _STATE["device"] = args.device

    install_import_stubs()
    install_robust_svd()
    sys.path.insert(0, str(POWDER_ROOT))

    m = load_manifest(args.manifest)
    ds = m["dataset"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    nb = m["protocol"]["nb_classes"]
    seed = m["seed"]
    inc, arrival, _ = compute_arrival()
    build_train_raw()
    load_base(args.data_root)
    build_datasets()

    n_tasks = nt if not args.limit_tasks else min(args.limit_tasks, nt)
    _STATE["n_tasks"] = n_tasks
    epochs_global = args.rounds * n_tasks

    ds_dir = "cifar100" if ds == "cifar100" else "ImageNet-R"
    ds_name = "CIFAR100" if ds == "cifar100" else "ImageNet_R"
    run_dir = (REPO_ROOT / "output" / ds_dir / "powder" / args.run_label
               / f"seed_{seed}")
    _STATE["run_dir"] = run_dir
    _STATE["training_log_dir"] = run_dir / "training_log"
    _STATE["training_log_dir"].mkdir(parents=True, exist_ok=True)
    _STATE["boundary_done"] = set()

    rec = metrics_recorder.MetricsRecorder(
        out_root=REPO_ROOT / "output", dataset=ds_dir, method="powder",
        seed=seed, run_name=args.run_label,
        meta={
            "manifest": str(Path(args.manifest).resolve()),
            "dataset": ds,
            "protocol": {"clients": nc, "tasks": nt,
                         "nb_classes": nb, "seed": seed},
            "adapter": "protocol_adapters/run_powder.py",
            "official_command": {
                "method": "cprompt_powder_v2", "prompt_flag": "codap_2d_v2",
                "prompt_param": [25, 10, 10, 8, 0, 0, 6, 10, 8],
                "num_clients": nc, "local_clients": nc,
                "epochs_local": args.epochs_local,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "global_update_lr": args.global_update_lr,
                "tasks_global": args.rounds,
                "epochs_global": epochs_global,
                "numclass": nb, "class_per_task": inc[0],
                "task_size": inc[0], "seed": seed},
            "arrival_increments": inc,
            "frozen_notes": [
                "class space = first-arrival order (positions)",
                "train data = client's own inner.train samples "
                "(persistent-slice analog); proportion=[0,1]",
                "eval pool = manifest test_pool",
                "full participation, no churn, fixed 5-client cohort",
                "uses per-round old-model ntd distillation; rotated "
                "public classes legally re-trained (no drop)",
                "sealed S1 (CodaPrompt_2d_v2 weight sizing -> "
                "e_task_number) and S2 (_ntd_loss refine_as_not_true2)",
            ],
            "limit_tasks": args.limit_tasks or None,
            "drops": "drops.json",
            "svd_rerouted_to_scipy": _STATE["svd_rerouted"],
        })
    _STATE["recorder"] = rec

    drops_path = run_dir / "drops.json"
    drops_path.write_text(json.dumps(_STATE["drops"], indent=2),
                          encoding="utf-8")

    # sealed runtime patches on the imported method classes (done BEFORE
    # fl_main imports them so every copy, incl. model_g, is patched)
    patch_codaprompt_init()
    patch_ntd_loss()

    # official CLI (option.args_parser consumes sys.argv inside the exec)
    sys.argv = [
        "fl_main_hard.py",
        "--dataset", ds_name,
        "--method", "cprompt_powder_v2",
        "--prompt_flag", "codap_2d_v2",
        "--prompt_param", *[str(x) for x in (25, 10, 10, 8, 0, 0, 6, 10, 8)],
        "--num_clients", str(nc),
        "--local_clients", str(nc),
        "--epochs_local", str(args.epochs_local),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.lr),
        "--global_update_lr", str(args.global_update_lr),
        "--tasks_global", str(args.rounds),
        "--epochs_global", str(epochs_global),
        "--numclass", str(nb),
        "--class_per_task", str(inc[0]),
        "--task_size", str(inc[0]),
        "--seed", str(seed),
        "--device", str(args.device),
    ]

    src = (POWDER_ROOT / "fl_main_hard.py").read_text("utf-8")
    src = patch_fl_main_source(src)

    ns = {
        "__name__": "__main__",
        "__file__": str(POWDER_ROOT / "fl_main_hard.py"),
        "_POWDER_STATE": _STATE,
        "_POWDER_GLOBAL_EVAL": _powder_global_eval,
        "_POWDER_MANIFEST_STRUCTURES": _powder_manifest_structures,
    }
    _STATE["ns"] = ns
    os.chdir(POWDER_ROOT)
    print(f"[powder-adapter] exec fl_main_hard.py: rounds/task="
          f"{args.rounds}, tasks={n_tasks}, epochs_global={epochs_global}, "
          f"inc={inc}, numclass={nb}, device={args.device}")
    exec(compile(src, str(POWDER_ROOT / "fl_main_hard.py"), "exec"), ns)

    rec.flush(n_tasks)
    print(f"[powder-adapter] DONE. output: {run_dir}")


if __name__ == "__main__":
    main()
