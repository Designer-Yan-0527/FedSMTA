"""FedProTIP (TMLR 2025, external_baselines/FedProTIP) training entry with
the frozen protocol manifest injected.

Frozen design (external-baseline fairness policy, same as FedMGP/C2Prompt):

  The official FedProTIP implementation is preserved verbatim; ONLY the
  dataset-partition interface is replaced so FedProTIP consumes the exact
  same client/task/class/sample assignment as FedSMTA. No file inside
  external_baselines/FedProTIP is ever edited; all patches are applied at
  runtime from this module.

Protocol reconciliation for a GLOBAL class-IL model (the core problem):

  FedSMTA's frozen split is per-client (8 classes per client-task). A
  single global model needs ONE label space, so classes are ordered by
  FIRST ARRIVAL: global task t's new-class set = union of the 5 clients'
  task-t masks, minus everything that arrived earlier. The manifest's
  CIFAR public classes ROTATE across tasks per client (25 public classes
  each appear in 2-4 different global tasks across their 5 client
  appearances), so arrival is genuinely non-uniform:

      CIFAR-100  new classes per task = [32, 26, 14, 16, 12]  (sum 100)
      ImageNet-R new classes per task = [40, 40, 40, 40, 40]  (sum 200)

  Raw labels are remapped into arrival order (train AND test targets),
  DataManager._increments is replaced by the arrival counts, and the
  per-task fc block width (IncrementalNet.generate_fc hardcodes
  args['increment']) follows the same counts. Because the head, the
  known/total bookkeeping and np.arange(known, total) all key off the
  arrival order, FedProTIP's own class-IL machinery (update_fc block
  freezing, GPM projection-space finetuning, task inference) runs
  natively on the frozen split with zero method modifications.

  Cross-task rotated public classes (CIFAR only): a client's task-t
  samples whose class already arrived at an earlier global task CANNOT
  legally enter training -- Client_FedProTIP._finetune computes
  fake_targets = labels - known_classes and trains only the new fc
  block (old labels would produce negative targets / CUDA asserts), and
  FedProTIP has no replay mechanism. Those samples are DROPPED from
  that client's local training set, the drop accounting is recorded in
  {run_dir}/drops.json, and each class is learned exactly once, at its
  arrival task, from the union of every client's samples of that class
  that arrive there. ImageNet-R is all-private (every class has exactly
  one owner client-task), so INR has zero drops and uniform 40-class
  tasks. Paper note for FedProTIP's CIFAR row: uses only first-arrival
  samples; no replay.

Injection points (all fail-loud, seal-checked by gate_fedprotip.py):

  1. utils.data_manager._map_new_class_index(y, order)
       -> identity class_order stays, but targets are remapped into
          ARRIVAL order (raw class -> first-arrival position).
  2. DataManager.__init__ wrapper
       -> after the official init: self._increments = arrival counts
          [32,26,14,16,12] / [40]*5 (drives nb_tasks, get_task_size,
          known/total, update_fc timing, native test blocks), and the
          test arrays are filtered to the manifest test_pool
          (test set minus the per-class surrogate samples FedSMTA
          consumes as surrogate-phase training data; C2Prompt used the
          same eval pool).
  3. utils.data_manager.partition_data(labels, beta, n_parties)
       -> replaced by the manifest assignment. The server's call site
          is surgically patched to pass cur_task + the DataManager so
          the splitter can resolve manifest positions. For client c at
          task t: positions of inner.train(c,t) samples whose class is
          NEW at t, inside the task-t new-class DummyDataset (which
          get_dataset(np.arange(K_{t-1}, K_t)) builds in a
          deterministic order). beta is ignored (the manifest IS the
          partition).
  4. IncrementalNet.generate_fc source patch
       -> the per-task fc block width args['increment'] becomes the
          per-task arrival count (4 occurrences inside the method).
  5. utils.data.data_dir = our local_datasets (CIFAR-100 already
     present, no download); Imagenet_R.download_data is replaced to
     enumerate OUR local_datasets/imagenet-r/{train,test} layout
     (sorted folders -> sorted files = the manifest raw_indices
     domain, same walk as data.continual_datasets.Imagenet_R in the
     main repo). The official downloader re-downloads every run
     (isfile() check on a directory) and splits via external txt
     lists, so it is unusable and is bypassed wholesale.
  6. Server_FedProTIP.train source patch (2 anchors)
       -> (a) partition_data call site passes cur_task/_dm;
          (b) after the native boundary evals, an RNG-neutral
          per-(client, test-task) eval records
          (client_id, after_task, test_task, accuracy) into the
          RunManager-format CSV tree via protocol_adapters/
          metrics_recorder.py. The eval loops the manifest test_pool
          restricted to each client's task-k class mask with logits
          masked to arrived classes, shuffle=False, model.eval(),
          no_grad -- zero RNG consumption, zero impact on training.
  7. wandb / progress stubs (neither is installed in the frozen env;
     wandb is disabled via args, progress.bar.Bar is imported but
     never used by the FedProTIP server/client code paths).

Protocol mapping (frozen, enforced against the manifest):
  5 clients x 5 tasks, full participation (frac=1.0), seed = manifest
  seed (42), classIL="True", class_shuffle=0 (identity class order;
  arrival remap does the reordering), 5 FL rounds per task (epochs,
  = the frozen comparison protocol), local_epochs=30 (frozen protocol;
  official default 5), eval pool = manifest test_pool.
Method-native (official configs/cifar100/fedprotip.json and
configs/imagenet-r/imagenet-r_fedprotip.json, overridable):
  net=resnet18, pretrained=True, lr=0.01 SGD, momentum=0, wd=5e-4,
  batch_size=64, threshold=0.775/0.7, global_proj="False",
  gt/use_pred_task/task-inference machinery untouched.
Memory/replay column: none (FedProTIP trains on current-task data
only; first-arrival drop documented above).

Usage (from the FedSMTA repo root, FedSMTA conda env):

  python protocol_adapters/run_fedprotip.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      [--data-root local_datasets] [--device cuda:0] \
      [--epochs 5] [--local-epochs 30] [--batch-size 64] \
      [--limit-tasks N]   # SMOKE ONLY: truncate to the first N tasks

Output: output/{cifar100|ImageNet-R}/fedprotip/manifest_adapter/
seed_{seed}/  (metrics/ CSV tree + meta.json + drops.json)
"""

import argparse
import inspect
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import metrics_recorder  # noqa: E402

FPT_ROOT = REPO_ROOT / "external_baselines" / "FedProTIP"

_STATE = {
    "manifest": None,
    "increments": None,      # new-class counts per task (arrival order)
    "arrival": None,         # raw class -> arrival position
    "new_at_task": None,     # t -> sorted list of new raw classes
    "train_raw": None,       # (c, t) -> inner.train raw trainset indices
    "drops": {},             # (c, t) -> number of dropped samples
    "data_root": None,
    "svd_rerouted": False,   # numpy LAPACK broken -> scipy reroute active
}

# ---------------------------------------------------------------------------
# sealed env bugfix: numpy LAPACK correctness self-test + scipy reroute
# ---------------------------------------------------------------------------


def install_robust_svd():
    """On some hosts (verified on this autodl box: numpy's OpenBLAS ILP64
    LAPACK) np.linalg.svd silently returns WRONG singular values (e.g.
    S_max=157 on a matrix whose Frobenius norm is 104.9; confirmed
    against a BLAS-free power iteration -- torch CPU/MKL and
    scipy.linalg are correct). FedProTIP's GPM subspace construction
    (client_fedprotip.py, 6 call sites of np.linalg.svd) consumes
    these numbers, so a broken LAPACK corrupts the method itself.

    Sealed-bugfix policy: fail-loud self-test at startup; if numpy SVD
    is broken, reroute np.linalg.svd -> scipy.linalg.svd FOR THIS
    PROCESS ONLY (official repo untouched; the seal check in
    gate_fedprotip.py asserts the official np.linalg.svd anchors are
    still present). On a healthy host (izfree) the reroute is skipped
    and the method runs with its native numpy calls.
    """
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
        # hermitian=True never occurs in this repo's call sites; the
        # scipy path below is the mathematically identical LAPACK call
        if compute_uv:
            return sla.svd(a, full_matrices=full_matrices)
        return sla.svd(a, full_matrices=full_matrices, compute_uv=False)

    np.linalg.svd = _robust_svd


# ---------------------------------------------------------------------------
# manifest / arrival-order bookkeeping
# ---------------------------------------------------------------------------


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema") != "fedsmta-split-manifest-v1":
        raise ValueError(f"{path}: unsupported schema {m.get('schema')!r}")
    if m.get("dataset") not in ("cifar100", "ImageNet-R"):
        raise ValueError(f"{path}: dataset {m.get('dataset')!r} not "
                         "supported by the FedProTIP adapter")
    p = m["protocol"]
    if (p["client_num"], p["task_num"]) != (5, 5):
        raise ValueError(f"{path}: protocol {p} does not match the frozen "
                         "FedProTIP alignment (5 clients x 5 tasks)")
    _STATE["manifest"] = m
    return m


def compute_arrival():
    """Global class-IL reconciliation: order classes by first arrival.

    new[t] = union of the 5 clients' task-t masks minus everything seen
    earlier. Labels are remapped so that new[0] occupy positions
    [0, n0), new[1] the next n1, etc. Returns (increments, arrival map,
    new_at_task) and caches them in _STATE.
    """
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


# ---------------------------------------------------------------------------
# import stubs (wandb / progress are not installed in the frozen env)
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
    if "progress" not in sys.modules:
        bar_mod = types.ModuleType("progress.bar")

        class _Bar:  # imported-but-unused by the FedProTIP code paths
            pass
        bar_mod.Bar = _Bar
        prog = types.ModuleType("progress")
        prog.bar = bar_mod
        sys.modules["progress"] = prog
        sys.modules["progress.bar"] = bar_mod
    os.environ.setdefault("MPLBACKEND", "Agg")
    # timm shim: the frozen env pins timm 0.6.7 (required by the main
    # repo's ViT lineage); FedProTIP's models/vit.py imports
    # `from timm.layers import resample_abs_pos_embed`, which only
    # exists in timm >= 0.9. The resnet18 path never CALLS it (vit.py's
    # own resize_pos_embed delegates to it only when loading a ViT
    # checkpoint), so a faithful port of the official timm>=0.9
    # implementation keeps the import honest without touching the env.
    if "timm.layers" not in sys.modules:
        try:
            import timm.layers  # noqa: F401
        except ImportError:
            import torch
            import torch.nn.functional as F

            def resample_abs_pos_embed(
                    posemb, new_size, old_size,
                    num_prefix_tokens=1, interpolation="bicubic",
                    antialias=True):
                """Port of timm.layers.pos_embed.resample_abs_pos_embed
                (timm >= 0.9); used only by vit.py's checkpoint loading
                in the ViT path."""
                new_ntok = new_size[0] * new_size[1]
                if not isinstance(posemb, torch.Tensor):
                    posemb = torch.tensor(posemb)
                new_posemb = posemb.reshape(
                    1, -1, -1) if posemb.ndim == 2 else posemb
                batched = new_posemb.shape[0] > 1
                if num_prefix_tokens:
                    posemb_prefix = (
                        new_posemb[:, :num_prefix_tokens]
                        if batched else new_posemb[:num_prefix_tokens])
                    new_posemb = (
                        new_posemb[:, num_prefix_tokens:]
                        if batched else new_posemb[num_prefix_tokens:])
                else:
                    posemb_prefix = new_posemb[:0]
                old_ntok = new_posemb.shape[1]
                if not old_size:
                    gs_old = int(math.sqrt(old_ntok))
                    old_size = (gs_old, gs_old)
                posemb_grid = new_posemb.reshape(
                    *[1, *old_size, -1] if not batched
                    else [new_posemb.shape[0], *old_size, -1]
                ).permute(0, 3, 1, 2)
                posemb_grid = F.interpolate(
                    posemb_grid, size=list(new_size), mode=interpolation,
                    align_corners=(interpolation in ("bilinear", "bicubic")),
                    antialias=antialias)
                posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
                    new_posemb.shape[0] if batched else 1, new_ntok, -1)
                posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
                return posemb.reshape(
                    [-1, new_ntok + num_prefix_tokens, new_posemb.shape[-1]]
                    if not batched else
                    [new_posemb.shape[0], new_ntok + num_prefix_tokens,
                     new_posemb.shape[-1]])

            import math  # noqa: F401  (used inside the port above)
            layers_mod = types.ModuleType("timm.layers")
            layers_mod.resample_abs_pos_embed = resample_abs_pos_embed
            sys.modules["timm.layers"] = layers_mod
            try:
                import timm
                timm.layers = layers_mod
            except ImportError:
                pass  # impossible: timm 0.6.7 exists in the frozen env


# ---------------------------------------------------------------------------
# dataset-level patches (utils.data / utils.data_manager)
# ---------------------------------------------------------------------------


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


def patch_data_module(data_root):
    """CIFAR reads our local_datasets (data_dir global); Imagenet_R's
    downloader is replaced with a pure-local walk of our layout."""
    import utils.data as udata

    dataset = _STATE["manifest"]["dataset"]
    if dataset == "cifar100":
        root = os.path.expanduser(data_root)
        if not os.path.isdir(os.path.join(root, "cifar-100-python")):
            raise RuntimeError(f"{root}: cifar-100-python not found")
        udata.data_dir = root
    else:  # ImageNet-R
        inr_root = os.path.join(os.path.expanduser(data_root), "imagenet-r")
        tr_d, tr_t = walk_imagenet_r(inr_root, "train")
        te_d, te_t = walk_imagenet_r(inr_root, "test")

        def manifest_download_data(self):
            self.train_data = np.array(tr_d)
            self.train_targets = np.array(tr_t)
            self.test_data = np.array(te_d)
            self.test_targets = np.array(te_t)

        udata.Imagenet_R.download_data = manifest_download_data
    _STATE["data_root"] = os.path.abspath(os.path.expanduser(data_root))


def patch_data_manager_module(increments):
    """Module-level patches on utils.data_manager:
    _map_new_class_index -> arrival remap (train+test targets)."""
    import utils.data_manager as dm

    arrival = _STATE["arrival"]
    manifest = _STATE["manifest"]
    test_pool = sorted(int(x) for x in manifest["test_pool"]["indices"])

    def manifest_map(y, order):  # replaces order.index-based identity
        return np.array([arrival[int(x)] for x in y])

    official_src = inspect.getsource(dm._map_new_class_index)
    if "order.index(x)" not in official_src:
        raise RuntimeError("utils/data_manager.py drift: "
                           "_map_new_class_index no longer maps through "
                           "class_order -- re-audit before patching")
    dm._map_new_class_index = manifest_map

    # DataManager.__init__ wrapper: official init, then _increments =
    # arrival counts + test-pool filter (both are the frozen protocol).
    original_init = dm.DataManager.__init__

    def manifest_init(self, dataset_name, shuffle, seed, init_cls,
                     increment, args):
        original_init(self, dataset_name, shuffle, seed, init_cls,
                      increment, args)
        keep = np.array(test_pool, dtype=np.int64)
        if keep.max() >= len(self._test_targets):
            raise RuntimeError("test_pool exceeds the official test set")
        self._test_data = self._test_data[keep]
        self._test_targets = self._test_targets[keep]
        self._increments = list(increments)

    dm.DataManager.__init__ = manifest_init


def manifest_partition_data(y_train, beta=0.4, n_parties=5,
                            cur_task=None, _dm=None):
    """Manifest replacement for utils.data_manager.partition_data.

    Maps each client's frozen inner.train(c,t) samples (restricted to
    classes that are NEW at global task t) to positions inside the
    task-t DummyDataset that get_dataset(np.arange(K_{t-1}, K_t))
    builds (ascending class positions; within a class, trainset
    order). Already-arrived-class samples are dropped (FedProTIP has
    no legal consumption path for old-class data at a new task) and
    counted in _STATE['drops'].
    """
    import numpy as np
    if cur_task is None or _dm is None:
        raise RuntimeError("partition_data called without the patched "
                           "call site (cur_task/_dm) -- the server source "
                           "patch is not active")
    inc = _STATE["increments"]
    k0 = int(sum(inc[:cur_task]))
    k1 = k0 + int(inc[cur_task])
    tgt = _dm._train_targets

    # position of every sample of the new classes inside the task-t
    # DummyDataset (offsets follow ascending arrival positions)
    pos_map = {}
    offset = 0
    for p in range(k0, k1):
        raws = np.where(tgt == p)[0]
        for i, r in enumerate(raws):
            pos_map[int(r)] = offset + i
        offset += len(raws)

    tr = _STATE["train_raw"]
    net = {}
    for cid in range(n_parties):
        lst, dropped = [], 0
        for r in tr[(cid, cur_task)]:
            lbl = int(tgt[r])
            if lbl < k0:
                dropped += 1      # class already arrived -> no legal path
            elif lbl >= k1:
                raise RuntimeError(
                    f"manifest violation: client {cid} task {cur_task} "
                    f"raw index {r} has arrival position {lbl} outside "
                    f"[{k0}, {k1}) -- task masks exceed the new-class set")
            else:
                lst.append(pos_map[r])
        net[cid] = np.array(lst, dtype=np.int64)
        _STATE["drops"][f"c{cid}t{cur_task}"] = dropped
    from utils.data_manager import record_net_data_stats
    stats = record_net_data_stats(y_train, net)
    return net, stats


def patch_inc_net_generate_fc(increments):
    """IncrementalNet.generate_fc: per-task block width must follow the
    arrival counts instead of the uniform args['increment']."""
    import inspect
    import textwrap
    from utils.inc_net import IncrementalNet

    src = textwrap.dedent(inspect.getsource(IncrementalNet.generate_fc))
    anchor = "torch.nn.Linear(in_dim, self.args['increment'])"
    n = src.count(anchor)
    if n != 3:  # classIL-pretrained / classIL-unpretrained / domain-t0;
        # the DR variant line carries bias=False and stays native (dead
        # under method="FedGPM")
        raise RuntimeError(
            f"utils/inc_net.py drift: expected 3 occurrences of the fc "
            f"width anchor inside IncrementalNet.generate_fc, found {n}; "
            "re-audit the official repo before patching")
    new = "torch.nn.Linear(in_dim, _FPT_STATE['increments'][_cur_task])"
    src = src.replace(anchor, new)
    import utils.inc_net as inc_net_mod
    ns = dict(vars(inc_net_mod))
    ns["_FPT_STATE"] = _STATE
    exec(compile(src, "<IncrementalNet.generate_fc manifest patch>",
                 "exec"), ns)
    IncrementalNet.generate_fc = ns["generate_fc"]


# ---------------------------------------------------------------------------
# server source patches (call site + boundary eval insertion)
# ---------------------------------------------------------------------------

TRAIN_SOURCE_PATCHES = [
    # (old, new, expected count)
    # partition_data call site: pass the task context + DataManager
    ("user_groups,_ = partition_data(train_dataset.labels, beta=self.args[\"beta\"], n_parties=self.args[\"n_clients\"])",
     "user_groups,_ = partition_data(train_dataset.labels, "
     "beta=self.args[\"beta\"], n_parties=self.args[\"n_clients\"], "
     "cur_task=self._cur_task, _dm=self.data_manager)",
     1),
    # RNG-neutral per-(client, test-task) boundary eval, recorded into
    # the RunManager-format CSV tree; runs BEFORE known_classes advances
    # (getsource + dedent strips the 4-space class indent: the anchor
    # lives at 8 spaces in the dedented method body)
    ("        self._known_classes = self._total_classes",
     "        _FPT_BOUNDARY_EVAL(self, task)\n"
     "        self._known_classes = self._total_classes",
     1),
]


def patch_server(server_mod):
    """Surgically patch Server_FedProTIP.train's source (runtime exec,
    zero repo-file edits) and bind the adapter helpers into the module."""
    import inspect
    import textwrap
    from server.server_fedprotip import Server_FedProTIP

    src = textwrap.dedent(inspect.getsource(Server_FedProTIP.train))
    for old, new, count in TRAIN_SOURCE_PATCHES:
        got = src.count(old)
        if got != count:
            raise RuntimeError(
                f"Server_FedProTIP.train source patch failed: expected "
                f"{count} occurrence(s) of {old[:60]!r}..., found {got}; "
                "re-audit the official repo before patching")
        src = src.replace(old, new)

    ns = dict(vars(server_mod))
    ns["_FPT_STATE"] = _STATE
    ns["_FPT_RECORDER"] = _STATE["recorder"]
    ns["_FPT_BOUNDARY_EVAL"] = _fpt_boundary_eval
    exec(compile(src, "<Server_FedProTIP.train manifest patch>", "exec"),
         ns)
    Server_FedProTIP.train = ns["train"]


def _fpt_boundary_eval(server, task):
    """RNG-neutral boundary eval: the global model on the manifest
    test_pool, per (client c, test task k<=task) restricted to client
    c's frozen task-k class mask, logits masked to arrived classes.
    Pure observation of the trained model; no RNG consumption."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    st = _STATE
    m = st["manifest"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    inc = st["increments"]
    arrived = int(sum(inc[:task + 1]))
    dm = server.data_manager
    arrival = st["arrival"]

    server.global_model.eval()
    bs = server.args["test_batch_size"]
    with torch.no_grad():
        for c in range(nc):
            for k in range(task + 1):
                mask = m["outer"][str(c)][str(k)]["class_mask"]
                positions = sorted(arrival[int(cl)] for cl in mask)
                ds = dm.get_dataset(positions, source="test", mode="test")
                if len(ds) == 0:
                    continue
                loader = DataLoader(ds, batch_size=bs, shuffle=False)
                correct = total = 0
                for _, x, y in loader:
                    x = x.to(server.device)
                    out = server.global_model(x)["logits"][:, :arrived]
                    pred = out.argmax(dim=1).cpu()
                    correct += int((pred.numpy() == y.numpy()).sum())
                    total += len(y)
                acc = 100.0 * correct / total if total else float("nan")
                st["recorder"].record(c, task, k, acc, phase="boundary")
    st["recorder"].flush(nt)


# ---------------------------------------------------------------------------
# args / config
# ---------------------------------------------------------------------------


def build_args(argv_parsed):
    dataset = _STATE["manifest"]["dataset"]
    seed = _STATE["manifest"]["seed"]
    if int(argv_parsed.seed) != int(seed):
        raise SystemExit(f"--seed {argv_parsed.seed} does not match the "
                         f"manifest seed {seed}: run a matching manifest, "
                         "not an off-protocol seed")

    cfg_rel = ("configs/cifar100/fedprotip.json" if dataset == "cifar100"
               else "configs/imagenet-r/imagenet-r_fedprotip.json")
    with open(FPT_ROOT / cfg_rel, "r", encoding="utf-8") as f:
        args = json.load(f)

    inc = _STATE["increments"]  # truncated iff --limit-tasks (smoke only)
    nb = _STATE["manifest"]["protocol"]["nb_classes"]

    # frozen protocol (hard)
    args["dataset"] = "cifar100" if dataset == "cifar100" else "imagenet-r"
    args["n_tasks"] = len(inc)
    args["n_classes"] = nb
    args["increment"] = inc[0]
    args["n_clients"] = 5
    args["frac"] = 1.0
    args["classIL"] = "True"
    args["class_shuffle"] = 0
    args["seed"] = int(seed)
    args["device"] = [argv_parsed.device]
    args["wandb"] = 0
    args["epochs"] = argv_parsed.epochs        # FL rounds per task (5)
    args["local_epochs"] = argv_parsed.local_epochs  # frozen: 30
    args["batch_size"] = argv_parsed.batch_size
    args["test_batch_size"] = args["batch_size"]
    args["method"] = "FedGPM"  # official dispatch name (no DR variant)
    return args


def main(argv=None):
    p = argparse.ArgumentParser(
        "FedProTIP + protocol adapter (frozen manifest split)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=str(REPO_ROOT / "output"))
    p.add_argument("--run-name", default="manifest_adapter")
    p.add_argument("--epochs", type=int, default=5,
                   help="FL rounds per task (frozen protocol: 5)")
    p.add_argument("--local-epochs", type=int, default=30,
                   help="local epochs per round (frozen protocol: 30; "
                        "official FedProTIP default: 5)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="official FedProTIP default: 64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-tasks", type=int, default=0,
                   help="SMOKE ONLY: truncate to the first N tasks "
                        "(off-protocol; formal runs must leave 0)")
    args_cli = p.parse_args(argv)

    install_import_stubs()
    install_robust_svd()
    load_manifest(args_cli.manifest)
    compute_arrival()
    if args_cli.limit_tasks:
        if not 0 < args_cli.limit_tasks < 5:
            raise SystemExit("--limit-tasks must be 1..4 (smoke only)")
        _STATE["increments"] = _STATE["increments"][:args_cli.limit_tasks]
    build_train_raw()

    # sys.path / cwd: FedProTIP root first (utils/models/server resolve
    # there), then its parent (the server files do
    # `from FedProTIP.client...` via sys.path.append('../') relative to
    # the cwd), then the adapter dir (metrics_recorder).
    fpt_root = str(FPT_ROOT.resolve())
    if fpt_root not in sys.path:
        sys.path.insert(0, fpt_root)
    parent = str(FPT_ROOT.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    os.chdir(fpt_root)

    recorder = metrics_recorder.MetricsRecorder(
        out_root=args_cli.out_root,
        dataset=_STATE["manifest"]["dataset"],
        method="fedprotip",
        seed=_STATE["manifest"]["seed"],
        run_name=args_cli.run_name,
        meta={
            "adapter": "run_fedprotip.py",
            "manifest": os.path.abspath(args_cli.manifest),
            "increments": _STATE["increments"],
            "new_at_task": _STATE["new_at_task"],
            "rounds_per_task": args_cli.epochs,
            "local_epochs": args_cli.local_epochs,
            "fracs": "full participation (frac=1.0)",
            "eval_pool": "manifest test_pool (surrogate-excluded)",
            "replay": "none (first-arrival drop for CIFAR rotated "
                      "public classes; see drops.json)",
            "numpy_lapack_selftest": (
                "BROKEN->scipy reroute" if _STATE["svd_rerouted"]
                else "healthy"),
            "limit_tasks": args_cli.limit_tasks or None,
        })
    _STATE["recorder"] = recorder

    # data-side patches, then the manager module, then the inc_net width
    patch_data_module(args_cli.data_root)
    patch_data_manager_module(_STATE["increments"])
    patch_inc_net_generate_fc(_STATE["increments"])

    import utils.data_manager as dm
    dm.partition_data = manifest_partition_data

    import server.server_fedprotip as server_mod
    # sanity: the server must bind OUR partition (imported after patch)
    if server_mod.partition_data is not manifest_partition_data:
        raise RuntimeError("server bound a different partition_data -- "
                           "patch ordering broken")
    patch_server(server_mod)

    args = build_args(args_cli)
    args["save_dir"] = str(recorder.run_dir)

    from server.server_fedprotip import Server_FedProTIP
    server = Server_FedProTIP(args)
    print("[protocol-adapter] FedProTIP run under the frozen manifest")
    print(f"[protocol-adapter] increments (new classes/task) = "
          f"{_STATE['increments']}")
    print(f"[protocol-adapter] metrics = {recorder.run_dir}")
    t0 = time.time()
    server.train()
    print(f"[protocol-adapter] train() done in "
          f"{(time.time() - t0) / 3600:.2f} h")
    drops = dict(sorted(_STATE["drops"].items()))
    with open(recorder.run_dir / "drops.json", "w",
              encoding="utf-8") as f:
        json.dump({"drops_per_client_task": drops,
                   "total_dropped": sum(drops.values())}, f, indent=2)
    print(f"[protocol-adapter] first-arrival drops: "
          f"{sum(drops.values())} samples -> drops.json")
    recorder.flush(args["n_tasks"])


if __name__ == "__main__":
    main()
