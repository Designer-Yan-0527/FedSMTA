"""Federated L2P / DualPrompt (fedl2p / feddualp) constructed baselines
under the frozen protocol manifest.

Both methods are single-machine continual-learning prompt baselines with
official standalone implementations (L2P: google-research flax repo under
external_baselines/L2P; DualPrompt: JH-LEE-KR timm repo under
external_baselines/DualPrompt). Following the constructed-baseline policy
of the frozen protocol family, this adapter FEDERATES them: the prompt
MODEL STRUCTURE is taken from the official CIFAR-100 configs verbatim,
while the entire training / aggregation / evaluation protocol is the
frozen one shared by run_fl_baseline.py and run_fedmgp.py.

Model construction:

  fedl2p   : the main repo's own L2P-style ViT
             (Models/vision_transformer__l2p.VisionTransformer -- the same
             class main.py itself builds with prompt_pool=True). Official
             L2P CIFAR-100 structure hyperparameters
             (external_baselines/L2P/configs/cifar100_l2p.py,
             prompt_pool_param block): pool_size=10, length=10, top_k=4,
             embedding_key='cls', prompt_key=True, batchwise_prompt=True,
             use_prompt_mask=False. L2P is task-agnostic: prompt selection
             is key-matching at BOTH train and eval time, which is exactly
             the forward(x) prompt_mask=None default path.
             head_type='token' (see "Deviations" below).
  feddualp : the main-line class has no G-prompt/E-prompt dual-pool
             support (no prefix tuning, no per-layer G/E split, no
             PreT_Attention), so the adapter borrows the OFFICIAL
             JH-LEE-KR DualPrompt model as-is: external_baselines/
             DualPrompt/vision_transformer.py is imported with its repo
             root temporarily on sys.path, registering its own
             vit_base_patch16_224 in the timm registry (timm 0.6.7:
             last import silently overwrites, so exactly ONE of the two
             model modules is imported per process, keyed on --method).
             Official DualPrompt CIFAR-100 structure hyperparameters
             (external_baselines/DualPrompt/configs/cifar100_dualprompt.py):
             G-prompt length 5 prefix-tuned on layers [0, 1]; E-prompt
             prompt-pool (pool_size=10, length=5, top_k=1) prefix-tuned on
             layers [2, 3, 4]; embedding_key='cls', prompt_key=True,
             batchwise_prompt=True, use_prompt_mask=True. Official
             semantics kept: local training on task t updates the
             dedicated E-prompt slot t*top_k (use_prompt_mask path),
             boundary eval is task-agnostic key-matching (forward(x)
             without task_id / train flag).

  Both models are built by the SAME
  create_model('vit_base_patch16_224', pretrained=True,
  pretrained_cfg=<pretrain_model/ViT-B_16.npz>) call as
  run_fl_baseline.build_model. The DualPrompt repo is timm-lineage and its
  _load_weights npz loader is byte-identical to the main repo's, so the
  offline npz bridge works on both routes with no wrapper and no network.

  A second plain ViT-B/16 (original_model: same npz weights, frozen
  forever, never aggregated) supplies the 'cls' prompt-query features
  (cls_features = original_model(x)['pre_logits']) -- the official
  original_model role of both repos (L2P train_continual.py L256, DualPrompt
  engine.py L49-56) and of the FedTA-lineage Client_DF.train (Models/
  Client_DF.py L239-242). Without it, embedding_key='cls' would silently
  fall back to max-pooling inside the prompt module.

Frozen-protocol口径 (identical to run_fl_baseline.py):

  - data: the frozen split manifest ONLY (outer client/task/class split
    + inner 70/30 per (client, task)). Local training uses the inner
    TRAIN split of the CURRENT task (federated sequential task stream,
    no replay, no old-task data); the inner TEST split is the eval pool;
  - one task = global_epoch rounds (default 5); every round every client
    starts from the current global weights and trains local_epoch epochs
    (default 30); Adam(lr, weight_decay=1e-3), CrossEntropyLoss, local
    batch 16 (repo-hardcoded Client_DF.train convention);
  - aggregation: sample-weighted FedAvg over the FULL state_dict. The
    backbone is frozen (requires_grad=False outside the prompt pool(s)
    and the head), so frozen keys are bit-identical across clients and
    averaging them is lossless (see aggregate_fedavg: keys identical
    across clients are copied bit-exactly instead of accumulated --
    mathematically the same weighted average of identical values).
    Trainable keys (prompt pool(s) + fixed nb_classes head) are plainly
    averaged -- NO protection mechanism of any kind;
  - evaluation: at every task boundary, every client is evaluated on its
    own inner TEST split of every SEEN task k, with logits masked to that
    (client, task)'s class_mask (same semantics as
    run_fedmgp._fedmgp_eval_task). R[t][k] lands in the RunManager-format
    CSV tree via metrics_recorder.MetricsRecorder.

Key differences vs the official single-machine L2P / DualPrompt (the
point of this baseline row):

  (a) the task stream is a FEDERATED sequential stream (all clients
      advance through the same global task index), not a single-machine
      stream;
  (b) after every round the prompt pool(s) AND the classification head
      are FedAvg'd with no protection -- this baseline exists to expose
      the aggregation-semantics defect of naively averaging a key-based
      prompt pool across clients;
  (c) the training schedule (optimizer / lr / epochs / batch / weight
      decay) is the frozen main-line protocol; ONLY the prompt structure
      hyperparameters come from the official configs;
  (d) no train-time class-mask on the logits (run_fl_baseline trains on
      the full fixed nb_classes head; official L2P / DualPrompt mask the
      logits to the current task during training). Boundary-eval masking
      is identical across all methods.

Deviations to be aware of:

  - head_type='token' for fedl2p: the official cifar100_l2p.py sets
    vit_classifier='prompt' (mean of the selected prompt outputs); this
    adapter uses the cls-token head, matching the head-input convention
    of every other baseline in the frozen family (deliberate, flagged).
  - official CIFAR-100 is a 10-task split; the frozen manifest is c5t5
    (5 tasks), so with the official DualPrompt pool_size=10 / top_k=1
    only E-prompt slots 0..4 receive task-masked training while slots
    5..9 stay at init (they remain key-matching candidates at eval).
    The official structure is kept verbatim on the 5-task stream.
  - the L2P / DualPrompt pull_constraint key-alignment term
    (loss - pull_coeff * reduce_sim, coeff 1.0 in both official CIFAR-100
    configs) is kept: it is part of the METHOD objective, not of the
    training schedule.

ImageNet-R manifests are rejected by an explicit assert for now (the
nb_classes=200 fixed-head route is a follow-up); this first version is
cifar100-only.

Usage (from the FedSMTA repo root):

  python protocol_adapters/run_prompt_baseline.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      --method {fedl2p|feddualp} \
      [--lr 0.001] [--local_epoch 30] [--global_epoch 5] \
      [--max_tasks 0] [--save_ckpt] [--resume auto] [--device cuda]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
DUALPROMPT_ROOT = REPO_ROOT / "external_baselines" / "DualPrompt"
for _p in (str(ADAPTER_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics_recorder  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from timm.models import create_model  # noqa: E402
from utils import build_transform  # noqa: E402

LOCAL_BATCH = 16  # repo-hardcoded local batch (official FedTA behavior)
WEIGHT_DECAY = 1e-03  # Client_DF.train convention
PULL_COEFF = 1.0  # official pull_constraint_coeff (both CIFAR-100 configs)
FROZEN_PROBE_KEY = "patch_embed.proj.weight"  # exists in both model classes

# Prompt STRUCTURE hyperparameters (the only thing taken from the official
# repos; every training hyperparameter comes from the frozen protocol).
L2P_STRUCTURE = dict(
    # source: external_baselines/L2P/configs/cifar100_l2p.py
    prompt_pool=True,           # config.prompt_pool
    pool_size=10,               # prompt_pool_param.pool_size
    prompt_length=10,           # prompt_pool_param.length
    top_k=4,                    # prompt_pool_param.top_k
    embedding_key="cls",        # prompt_pool_param.embedding_key
    prompt_key=True,            # prompt_pool_param.prompt_key
    prompt_init="uniform",      # prompt_pool_param.initializer
    prompt_key_init="uniform",  # prompt_pool_param.prompt_key_init
    batchwise_prompt=True,      # prompt_pool_param.batchwise_prompt
    use_prompt_mask=False,      # prompt_pool_param.use_prompt_mask
    head_type="token",          # frozen-family head convention (see docstring)
)
DUALPROMPT_STRUCTURE = dict(
    # source: external_baselines/DualPrompt/configs/cifar100_dualprompt.py
    prompt_pool=True,           # --prompt_pool
    pool_size=10,               # --size
    prompt_length=5,            # --length
    top_k=1,                    # --top_k
    embedding_key="cls",        # --embedding_key
    prompt_key=True,            # --prompt_key
    prompt_init="uniform",      # official main.py maps initializer->prompt_init
    prompt_key_init="uniform",  # --prompt_key_init
    batchwise_prompt=True,      # --batchwise_prompt
    use_prompt_mask=True,       # --use_prompt_mask (official DualPrompt)
    head_type="token",          # --head_type
    use_g_prompt=True,                       # --use_g_prompt
    g_prompt_length=5,                       # --g_prompt_length
    g_prompt_layer_idx=[0, 1],               # --g_prompt_layer_idx
    use_prefix_tune_for_g_prompt=True,       # --use_prefix_tune_for_g_prompt
    use_e_prompt=True,                       # --use_e_prompt
    e_prompt_layer_idx=[2, 3, 4],            # --e_prompt_layer_idx
    use_prefix_tune_for_e_prompt=True,       # --use_prefix_tune_for_e_prompt
    same_key_value=False,                    # --same_key_value
)
STRUCTURES = {"fedl2p": L2P_STRUCTURE, "feddualp": DUALPROMPT_STRUCTURE}

# Everything outside these prefixes is frozen (requires_grad=False):
# fedl2p   -> Global_Prompt module 'prompt' + fixed head
# feddualp -> 'g_prompt' parameter + EPrompt module 'e_prompt' + fixed head
TRAINABLE_PREFIXES = {
    "fedl2p": ("prompt.", "head."),
    "feddualp": ("g_prompt", "e_prompt.", "head."),
}


# --------------------------------------------------------------------------
# data: manifest -> per-(client, task) train/test subsets + class masks
# --------------------------------------------------------------------------

def build_client_data(manifest, data_path):
    """Returns data[client][task] = dict(train=Subset, test=Subset,
    class_mask=list, n_train=int). Both splits share the manifest outer
    subset (built with the TRAIN transform, as in the FedTA lineage: the
    inner test loaders in Client_DF.get_data are random_split out of the
    train-transformed outer subset, and build_transform(True, 224) is
    deterministic: Resize only). Identical to run_fl_baseline.
    build_client_data, cifar100-only."""
    name = manifest["dataset"]
    p = manifest["protocol"]
    if name != "cifar100":
        raise ValueError(f"unsupported manifest dataset {name!r} (this "
                         "adapter is cifar100-only in its first version)")
    trans = build_transform(True, p["input_size"])

    from torchvision.datasets import CIFAR100
    from data.cifar100_subset_spliter import CustomedSubset

    def wrap(idxs):
        base = CIFAR100(root=data_path, train=True, download=False)
        return CustomedSubset(base, list(idxs), trans)

    data = []
    for cid in range(p["client_num"]):
        per = []
        for tid in range(p["task_num"]):
            outer = manifest["outer"][str(cid)][str(tid)]
            inner = manifest["inner"][str(cid)][str(tid)]
            sub = wrap(outer["raw_indices"])
            train_sub = Subset(sub, list(inner["train"]))
            test_sub = Subset(sub, list(inner["test"]))
            per.append({
                "train": train_sub,
                "test": test_sub,
                "class_mask": list(outer["class_mask"]),
                "n_train": len(train_sub),
            })
        data.append(per)
    return data


# --------------------------------------------------------------------------
# model: method-specific registry setup + prompt ViT + frozen key backbone
# --------------------------------------------------------------------------

def setup_model_registry(method):
    """Imports the method's model module so its vit_base_patch16_224
    registration wins the timm registry (timm 0.6.7: the last import
    silently overwrites the entrypoint). The two methods ship DIFFERENT
    VisionTransformer classes under the same registry name, so exactly one
    of the two modules may be imported per process. Returns the model
    source path (for logs / meta)."""
    if method == "fedl2p":
        import Models.vision_transformer__l2p  # noqa: E402,F401
        return "Models/vision_transformer__l2p.py"
    # feddualp: borrow the official JH-LEE-KR DualPrompt model. Its
    # vision_transformer.py imports the top-level modules 'prompt' and
    # 'attention' from its own repo root, so the root goes onto sys.path
    # for the duration of the import and is removed afterwards (the
    # DualPrompt root also contains utils.py / models.py / datasets.py,
    # whose names must NOT leak onto the import path for the rest of the
    # process).
    root = str(DUALPROMPT_ROOT.resolve())
    sys.path.insert(0, root)
    try:
        import vision_transformer  # noqa: E402,F401 (DualPrompt's)
    finally:
        if root in sys.path:
            sys.path.remove(root)
    return "external_baselines/DualPrompt/vision_transformer.py"


def build_models(method, nb_classes, device):
    """Returns (original_model, model), both pretrained from the offline
    pretrain_model/ViT-B_16.npz exactly like run_fl_baseline.build_model.

    original_model: plain ViT-B/16 (no prompts), frozen forever and never
    aggregated; supplies the 'cls' prompt-query key features.

    model: the method's prompt model, built from whichever
    vit_base_patch16_224 registration setup_model_registry(method)
    installed, with the official CIFAR-100 structure hyperparameters."""
    base_cfg = create_model("vit_base_patch16_224").default_cfg

    def cfg():
        c = dict(base_cfg)
        c["file"] = str(REPO_ROOT / "pretrain_model" / "ViT-B_16.npz")
        return c

    original_model = create_model(
        "vit_base_patch16_224",
        pretrained=True,
        pretrained_cfg=cfg(),
        num_classes=nb_classes,
    )
    for prm in original_model.parameters():
        prm.requires_grad = False
    original_model.to(device)
    original_model.eval()

    model = create_model(
        "vit_base_patch16_224",
        pretrained=True,
        pretrained_cfg=cfg(),
        num_classes=nb_classes,
        **STRUCTURES[method],
    )
    model.to(device)
    return original_model, model


def freeze_and_audit(model, method):
    """Freezes every parameter outside TRAINABLE_PREFIXES[method] (i.e.
    everything except the prompt pool(s) and the classification head:
    patch_embed / blocks / norm / cls_token / pos_embed / fc_norm), then
    prints the full frozen/trainable manifest (name + shape + numel) for
    manual auditing."""
    prefixes = TRAINABLE_PREFIXES[method]
    for prm in model.parameters():
        prm.requires_grad = False
    n_train = n_frozen = 0
    numel_train = numel_frozen = 0
    for name, prm in model.named_parameters():
        trainable = name.startswith(prefixes)
        prm.requires_grad = trainable
        tag = "TRAINABLE" if trainable else "FROZEN"
        print(f"[param-audit] {tag:9s} {name:55s} "
              f"{str(tuple(prm.shape)):26s} numel={prm.numel()}")
        if trainable:
            n_train += 1
            numel_train += prm.numel()
        else:
            n_frozen += 1
            numel_frozen += prm.numel()
    if n_train == 0:
        raise RuntimeError(f"{method}: no trainable parameters matched "
                           f"prefixes {prefixes}; audit the model class")
    print(f"[param-audit] trainable tensors {n_train} "
          f"({numel_train} params) / frozen tensors {n_frozen} "
          f"({numel_frozen} params)")


# --------------------------------------------------------------------------
# local training + aggregation
# --------------------------------------------------------------------------

def local_train(model, original_model, global_sd, train_sub, task, lr,
                local_epoch, device, num_workers):
    """Trains `model` (already holding w_global) on train_sub for
    local_epoch epochs. Only requires_grad parameters enter the optimizer
    (prompt pool(s) + head). Returns (steps_taken, mean_loss).

    Forward convention = official L2P / DualPrompt engines and FedTA
    Client_DF.train: cls_features from the frozen original ViT; task_id
    passed so feddualp's use_prompt_mask path trains the dedicated
    E-prompt slot (fedl2p ignores task_id: use_prompt_mask=False). The
    pull_constraint term (official coeff 1.0) aligns prompt keys with
    queries and is part of the method objective."""
    model.load_state_dict(global_sd)
    model.train()
    original_model.eval()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    loader = DataLoader(train_sub, batch_size=LOCAL_BATCH, shuffle=True,
                        num_workers=num_workers)
    steps = 0
    loss_sum, loss_n = 0.0, 0
    for _ in range(local_epoch):
        for input_, target in loader:
            input_ = input_.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                cls_features = original_model(input_)["pre_logits"]
            res = model(input_, task_id=task, cls_features=cls_features,
                        train=True)
            loss = criterion(res["logits"], target) \
                - PULL_COEFF * res["reduce_sim"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach())
            loss_n += 1
    return steps, (loss_sum / max(1, loss_n))


def aggregate_fedavg(global_sd, client_sds, n_list):
    """Sample-weighted FedAvg over the FULL state_dict (the fedavg rule of
    run_fl_baseline.aggregate, no method-specific aggregation).

    Bit-exactness note: frozen keys are bit-identical across clients (they
    never receive gradients), so for any key where every client tensor
    equals the first one, the weighted average is replaced by a direct
    copy of the pre-round global tensor -- mathematically the same value
    (the weighted average of identical values), but computed bit-exactly:
    a float accumulation of w_i * v with sum(w_i) == 1 is NOT guaranteed
    to return v bit-for-bit. Trainable keys (prompt pool(s), head) always
    go through the weighted average."""
    n_tot = float(sum(n_list))
    w = [n / n_tot for n in n_list]
    new_sd = {}
    for k in global_sd:
        ref = client_sds[0][k]
        if all(torch.equal(sd[k], ref) for sd in client_sds[1:]):
            new_sd[k] = global_sd[k].detach().clone()
            continue
        acc = None
        for wi, sd in zip(w, client_sds):
            term = sd[k].float() * wi
            acc = term if acc is None else acc + term
        new_sd[k] = acc.to(global_sd[k].dtype)
    return new_sd


# --------------------------------------------------------------------------
# evaluation (same mask semantics as run_fedmgp._fedmgp_eval_task)
# --------------------------------------------------------------------------

def evaluate_task(model, original_model, test_sub, class_mask, nb_classes,
                  device, num_workers):
    """Task-agnostic boundary evaluation: forward WITHOUT task_id and with
    train=False, so both methods select prompts by key matching (the
    official eval path of L2P and DualPrompt); logits are then masked to
    the (client, task) class_mask."""
    model.eval()
    original_model.eval()
    loader = DataLoader(test_sub, batch_size=LOCAL_BATCH, shuffle=False,
                        num_workers=num_workers)
    not_mask = np.setdiff1d(np.arange(nb_classes), class_mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for input_, target in loader:
            input_ = input_.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            cls_features = original_model(input_)["pre_logits"]
            logits = model(input_, cls_features=cls_features)["logits"]
            logits = logits.index_fill(dim=1, index=not_mask,
                                       value=float("-inf"))
            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += int((predicts == target.cpu()).sum())
            total += len(target)
    return 100.0 * correct / max(1, total)


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------

def save_resume(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def rng_snapshot():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
        "np": np.random.get_state(),
        "py": random.getstate(),
    }


def rng_restore(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["np"])
    random.setstate(s["py"])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    cli = argparse.ArgumentParser(
        "Federated L2P / DualPrompt constructed baselines on the frozen "
        "protocol manifest")
    cli.add_argument("--manifest", required=True)
    cli.add_argument("--method", required=True,
                     choices=["fedl2p", "feddualp"])
    cli.add_argument("--lr", type=float, default=0.001)
    cli.add_argument("--local_epoch", type=int, default=30)
    cli.add_argument("--global_epoch", type=int, default=5,
                     help="rounds per task (frozen protocol value: 5)")
    cli.add_argument("--max_tasks", type=int, default=0,
                     help="truncate after N tasks (0 = all; smoke only)")
    cli.add_argument("--data_path", default="./local_datasets")
    cli.add_argument("--output_dir", default="output")
    cli.add_argument("--num_workers", type=int, default=2)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--save_ckpt", action="store_true",
                     help="save task_XX_end.pth (350MB each)")
    cli.add_argument("--resume", default="",
                     help="'' = fresh; 'auto' = run_dir/checkpoints/latest.pth")
    cli.add_argument("--run_name", default="",
                     help="'' = {method}_cifar (e.g. fedl2p_cifar)")
    args = cli.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != "fedsmta-split-manifest-v1":
        raise SystemExit(f"{args.manifest}: not a fedsmta-split-manifest-v1")
    p = manifest["protocol"]
    seed = int(manifest["seed"])
    dataset = manifest["dataset"]
    if dataset != "cifar100":
        raise SystemExit(
            f"{args.manifest}: dataset {dataset!r} is not supported by "
            "this first version (cifar100-only; the ImageNet-R "
            "nb_classes=200 fixed-head route is a follow-up)")
    client_num, task_num = p["client_num"], p["task_num"]
    nb_classes = p["nb_classes"]
    assert nb_classes == 100, \
        f"cifar100 manifest must have nb_classes=100, got {nb_classes}"
    device = torch.device(args.device)
    run_name = args.run_name or f"{args.method}_cifar"

    torch.backends.cudnn.benchmark = True  # official behavior (non-det.)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    structure = STRUCTURES[args.method]
    model_source = setup_model_registry(args.method)

    recorder = metrics_recorder.MetricsRecorder(
        out_root=REPO_ROOT / args.output_dir, dataset=dataset,
        method=args.method, seed=seed, run_name=run_name,
        meta={
            "manifest": str(Path(args.manifest).resolve()),
            "adapter": "protocol_adapters/run_prompt_baseline.py",
            "construction": "federated L2P / DualPrompt constructed "
                            "baselines under the frozen protocol manifest: "
                            "ViT-B/16 backbone frozen, prompt pool(s) + "
                            "fixed nb_classes head trained locally and "
                            "FedAvg'd globally with NO protection "
                            "mechanism (the row exists to expose the "
                            "aggregation-semantics defect of naively "
                            "averaging a key-based prompt pool)",
            "model_source": model_source,
            "prompt_structure": dict(structure),
            "structure_source": (
                "external_baselines/L2P/configs/cifar100_l2p.py"
                if args.method == "fedl2p" else
                "external_baselines/DualPrompt/configs/"
                "cifar100_dualprompt.py"),
            "trainable_parameters": list(TRAINABLE_PREFIXES[args.method]),
            "key_features": "frozen pretrained ViT-B/16 (original_model, "
                            "never aggregated) pre_logits as the 'cls' "
                            "prompt-query key -- the official "
                            "original_model role",
            "local_optimizer": f"Adam(lr={args.lr}, "
                               f"weight_decay={WEIGHT_DECAY})",
            "local_loss": f"CrossEntropy - {PULL_COEFF} * reduce_sim "
                          "(official pull_constraint, coeff 1.0 in both "
                          "official CIFAR-100 configs)",
            "local_batch": LOCAL_BATCH,
            "local_epoch": args.local_epoch,
            "global_epoch": args.global_epoch,
            "aggregation": "sample-weighted FedAvg over the full "
                           "state_dict; keys bit-identical across clients "
                           "(the frozen backbone) are copied bit-exactly "
                           "(= the weighted average of identical values)",
            "differences_vs_official": [
                "(a) federated sequential task stream, not single-machine",
                "(b) prompt pool(s) and head FedAvg'd after every round "
                "with no protection mechanism",
                "(c) training schedule is the frozen main-line protocol; "
                "only prompt structure hyperparameters are official",
                "(d) no train-time class-mask on the logits (official "
                "L2P/DualPrompt mask to the current task at train time)",
                "fedl2p head_type='token' (official cifar100_l2p.py uses "
                "vit_classifier='prompt')",
                "official 10-task CIFAR-100 vs frozen 5-task manifest: "
                "with pool_size=10/top_k=1 only E-prompt slots 0..4 "
                "receive task-masked training (feddualp)",
            ],
            "eval_pool": "per-(client, task) inner TEST split of the "
                         "manifest, logits masked to the (client, task) "
                         "class_mask, task-agnostic key-matching prompt "
                         "selection (same semantics as "
                         "run_fedmgp._fedmgp_eval_task)",
        })
    ckpt_dir = recorder.run_dir / "checkpoints"
    latest = ckpt_dir / "latest.pth"

    print("=" * 70)
    print(f"[fl-prompt] method      = {args.method}")
    print(f"[fl-prompt] model       = {model_source}")
    print(f"[fl-prompt] manifest    = {args.manifest}")
    print(f"[fl-prompt] dataset     = {dataset} (nb_classes={nb_classes})")
    print(f"[fl-prompt] clients     = {client_num}, tasks = {task_num}")
    print(f"[fl-prompt] lr={args.lr} local_epoch={args.local_epoch} "
          f"global_epoch={args.global_epoch} batch={LOCAL_BATCH} "
          f"pull_coeff={PULL_COEFF}")
    for key in sorted(structure):
        print(f"[fl-prompt] structure   = {key}: {structure[key]}")
    print(f"[fl-prompt] metrics     = {recorder.run_dir}")
    print("=" * 70)

    data = build_client_data(manifest, args.data_path)
    original_model, model = build_models(args.method, nb_classes, device)
    freeze_and_audit(model, args.method)

    # resume state: (task, round_in_task) at round boundaries
    start_task, start_round = 0, 0
    if args.resume == "auto":
        if not latest.is_file():
            print("[fl-prompt] --resume auto but no latest.pth; fresh run")
        else:
            # map_location must be "cpu": map_location=device would move the
            # saved CPU RNG-state ByteTensor to CUDA, and torch.set_rng_state
            # only accepts a CPU ByteTensor (model weights are still copied
            # cross-device by load_state_dict below).
            st = torch.load(latest, map_location="cpu")
            model.load_state_dict(st["model"])
            rng_restore(st["rng"])
            start_task, start_round = st["task"], st["round_in_task"]
            print(f"[fl-prompt] resumed at task {start_task} "
                  f"round {start_round}")
    elif args.resume:
        raise SystemExit("--resume only accepts '' or 'auto'")

    t_limit = task_num if args.max_tasks <= 0 else min(args.max_tasks,
                                                       task_num)
    for task in range(start_task, t_limit):
        r0 = start_round if task == start_task else 0
        for rnd in range(r0, args.global_epoch):
            t0 = time.time()
            global_sd = {k: v.detach().clone()
                         for k, v in model.state_dict().items()}
            client_sds, n_list = [], []
            for cid in range(client_num):
                steps, mean_loss = local_train(
                    model, original_model, global_sd,
                    data[cid][task]["train"], task, args.lr,
                    args.local_epoch, device, args.num_workers)
                client_sds.append({k: v.detach().clone() for k, v in
                                   model.state_dict().items()})
                n_list.append(data[cid][task]["n_train"])
                print(f"[fl-prompt] task {task} round {rnd} client {cid} "
                      f"loss {mean_loss:.4f} steps {steps}")
            new_sd = aggregate_fedavg(global_sd, client_sds, n_list)
            if task == 0 and rnd == 0:
                # frozen-param bitwise spot check: FedAvg must leave every
                # frozen parameter bit-identical to the pre-round global
                # value (which is what every client still holds)
                assert torch.equal(new_sd[FROZEN_PROBE_KEY],
                                   global_sd[FROZEN_PROBE_KEY]), (
                    f"frozen parameter {FROZEN_PROBE_KEY!r} changed "
                    "through FedAvg aggregation")
                print(f"[fl-prompt] frozen-param bitwise check OK "
                      f"({FROZEN_PROBE_KEY})")
            model.load_state_dict(new_sd)
            save_resume({"task": task, "round_in_task": rnd + 1,
                         "model": model.state_dict(), "rng": rng_snapshot()},
                        latest)
            print(f"[fl-prompt] task {task} round {rnd} aggregated "
                  f"({time.time() - t0:.0f}s)")

        # task-boundary full-matrix eval: every client x every seen task
        for cid in range(client_num):
            for k in range(task + 1):
                acc = evaluate_task(model, original_model,
                                    data[cid][k]["test"],
                                    data[cid][k]["class_mask"], nb_classes,
                                    device, args.num_workers)
                recorder.record(cid, task, k, acc, phase="boundary")
                print(f"[fl-prompt] client {cid} after_task {task} "
                      f"test_task {k} acc {acc:.2f}")
        recorder.flush(task_num)
        if args.save_ckpt:
            save_resume({"task": task + 1, "round_in_task": 0,
                         "model": model.state_dict(), "rng": rng_snapshot()},
                        ckpt_dir / f"task_{task:02d}_end.pth")

    print("[fl-prompt] done.")


if __name__ == "__main__":
    main()
