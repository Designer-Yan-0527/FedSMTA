"""Constructed FL-only baselines (FedAvg / FedProx / FedNova) under the
frozen protocol manifest.

These three baselines have NO official standalone implementation in the
FedTA lineage (FedTA arXiv 2412.18355 Table 1 reports them as constructed
baselines), so unlike run_fedmgp.py / run_c2p.py (which execute an
external official repo) this adapter IMPLEMENTS the methods directly in
the FedSMTA codebase, reusing its model factory, transforms and manifest
splits. Paper wording: "constructed following the FedTA paper
(arXiv 2412.18355) Table 1 setting".

Construction (identical for all three, only the local objective and the
aggregation rule differ):

  - backbone: ViT-B/16 ImageNet-pretrained (pretrain_model/ViT-B_16.npz),
    same create_model call as the main repo's original_model, with a
    fixed nb_classes head (100 cifar100 / 200 ImageNet-R) -> class-IL;
  - FULL-PARAMETER federated fine-tuning (no frozen blocks, no prompts,
    no anchors): the standard FL construction in the FedTA Table 1
    setting (FedAvg T1 ~ 50 on CIFAR-100, near-total late collapse);
  - data: the frozen split manifest ONLY (outer client/task/class split
    + inner 70/30 per (client, task)). Local training uses the inner
    TRAIN split of the CURRENT task (sequential task stream, no replay,
    no old-task data); the inner TEST split is the evaluation pool;
  - local optimizer: Adam(lr, weight_decay=1e-3), CrossEntropyLoss,
    local batch 16 -- the same local-training convention as the FedTA
    lineage Client_DF.train (repo-hardcoded batch 16 included);
  - one task = global_epoch rounds; every round every client trains
    local_epoch epochs from the current global weights (100% partic.).

Method-specific rules:

  fedavg : w <- sum_i (n_i / sum_j n_j) * w_i
           (sample-weighted average, the canonical FedAvg objective)
  fedprox: local loss += (mu/2) * ||w_i - w_global||^2 (all params,
           w_global snapshotted at round start); aggregation = fedavg
  fednova: d_i = w_i - w (raw local update), a_i = tau_i = optimizer
           steps actually taken (= local_epoch * ceil(n_i/16));
           a_eff = sum_i (n_i/sum n_j) * a_i;
           w <- w + a_eff * sum_i (n_i/sum n_j) * d_i / a_i
           (data-weighted mean of per-step-normalized updates, scaled by
           the data-weighted mean step count: normalize-then-reweight,
           Wang et al. NeurIPS 2020 Algorithm 1 with the server step set
           to the mean local step count; degenerates EXACTLY to
           sample-weighted FedAvg when all a_i are equal).

Evaluation (identical semantics to run_fedmgp._fedmgp_eval_task and the
main repo's Client_DF.evaluate): at every task boundary, every client is
evaluated on its own inner TEST split of every SEEN task k, with logits
masked to that (client, task)'s class_mask. R[t][k] lands in the
RunManager-format CSV tree via metrics_recorder.MetricsRecorder, so all
methods share one three-level (dataset / seed / task) aggregation.

Usage (from the FedSMTA repo root):

  python protocol_adapters/run_fl_baseline.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      --method {fedavg|fedprox|fednova} [--fedprox_mu 0.01] \
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
for _p in (str(ADAPTER_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics_recorder  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import Models.vision_transformer__l2p  # noqa: E402,F401 (registers vit_base_patch16_224)
from timm.models import create_model  # noqa: E402
from utils import build_transform  # noqa: E402

LOCAL_BATCH = 16  # repo-hardcoded local batch (official FedTA behavior)
WEIGHT_DECAY = 1e-03  # Client_DF.train convention


# --------------------------------------------------------------------------
# data: manifest -> per-(client, task) train/test subsets + class masks
# --------------------------------------------------------------------------

def build_client_data(manifest, data_path):
    """Returns data[client][task] = dict(train=Subset, test=Subset,
    class_mask=list, n_train=int). Both splits share the manifest outer
    subset (built with the TRAIN transform, as in the FedTA lineage: the
    inner test loaders in Client_DF.get_data are random_split out of the
    train-transformed outer subset, and build_transform(True, 224) is
    deterministic: Resize only)."""
    name = manifest["dataset"]
    p = manifest["protocol"]
    trans = build_transform(True, p["input_size"])

    if name == "cifar100":
        from torchvision.datasets import CIFAR100
        from data.cifar100_subset_spliter import CustomedSubset

        def wrap(idxs):
            base = CIFAR100(root=data_path, train=True, download=False)
            return CustomedSubset(base, list(idxs), trans)
    elif name == "ImageNet-R":
        from data.continual_datasets import Imagenet_R
        from data.imagenet_r_subset_spliter import CustomedSubset

        base = Imagenet_R(root=data_path, train=True, download=False)

        def wrap(idxs):
            return CustomedSubset(base, list(idxs), trans, None)
    else:
        raise ValueError(f"unsupported manifest dataset {name!r}")

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
# model: plain ViT-B/16, full fine-tuning, fixed nb_classes head
# --------------------------------------------------------------------------

def build_model(nb_classes, device):
    pretrained_cfg = create_model("vit_base_patch16_224").default_cfg
    pretrained_cfg["file"] = str(REPO_ROOT / "pretrain_model" / "ViT-B_16.npz")
    model = create_model(
        "vit_base_patch16_224",
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        num_classes=nb_classes,
    )
    return model.to(device)


# --------------------------------------------------------------------------
# local training + aggregation
# --------------------------------------------------------------------------

def local_train(model, global_sd, train_sub, lr, local_epoch, device,
                num_workers, mu=0.0):
    """Trains `model` (already holding w_global) on train_sub for
    local_epoch epochs. Returns (steps_taken, mean_ce_loss)."""
    model.load_state_dict(global_sd)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    loader = DataLoader(train_sub, batch_size=LOCAL_BATCH, shuffle=True,
                        num_workers=num_workers)
    steps = 0
    loss_sum, loss_n = 0.0, 0
    for _ in range(local_epoch):
        for input_, target in loader:
            input_ = input_.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(input_)["logits"]
            loss = criterion(logits, target)
            if ref is not None:
                prox = sum((p - g).pow(2).sum()
                           for p, g in zip(model.parameters(), ref))
                loss = loss + 0.5 * mu * prox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach())
            loss_n += 1
    return steps, (loss_sum / max(1, loss_n))


def aggregate(method, global_sd, client_sds, n_list, a_list):
    """Returns the new global state_dict.

    fedavg/fedprox: sample-weighted average of the w_i.
    fednova:        w + a_eff * sum_i (n_i/sum n) * (w_i - w)/a_i,
                    a_eff = sum_i (n_i/sum n) * a_i.
    """
    n_tot = float(sum(n_list))
    w = [n / n_tot for n in n_list]
    new_sd = {}
    if method in ("fedavg", "fedprox"):
        for k in global_sd:
            acc = None
            for wi, sd in zip(w, client_sds):
                term = sd[k].float() * wi
                acc = term if acc is None else acc + term
            new_sd[k] = acc.to(global_sd[k].dtype)
    elif method == "fednova":
        a_eff = sum((n / n_tot) * a for n, a in zip(n_list, a_list))
        coeff = [(n / n_tot) * (a_eff / a)
                 for n, a in zip(n_list, a_list)]
        for k in global_sd:
            delta = None
            for c, sd in zip(coeff, client_sds):
                term = (sd[k].float() - global_sd[k].float()) * c
                delta = term if delta is None else delta + term
            new_sd[k] = (global_sd[k].float() + delta).to(global_sd[k].dtype)
    else:
        raise ValueError(f"unknown method {method!r}")
    return new_sd


# --------------------------------------------------------------------------
# evaluation (same mask semantics as run_fedmgp._fedmgp_eval_task)
# --------------------------------------------------------------------------

def evaluate_task(model, test_sub, class_mask, nb_classes, device,
                  num_workers):
    model.eval()
    loader = DataLoader(test_sub, batch_size=LOCAL_BATCH, shuffle=False,
                        num_workers=num_workers)
    not_mask = np.setdiff1d(np.arange(nb_classes), class_mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for input_, target in loader:
            input_ = input_.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(input_)["logits"]
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
        "FL-only constructed baselines (FedAvg/FedProx/FedNova) on the "
        "frozen protocol manifest")
    cli.add_argument("--manifest", required=True)
    cli.add_argument("--method", required=True,
                     choices=["fedavg", "fedprox", "fednova"])
    cli.add_argument("--fedprox_mu", type=float, default=0.01,
                     help="FedProx proximal coefficient (main-table 0.01)")
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
    cli.add_argument("--run_name", default="manifest_adapter")
    args = cli.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != "fedsmta-split-manifest-v1":
        raise SystemExit(f"{args.manifest}: not a fedsmta-split-manifest-v1")
    p = manifest["protocol"]
    seed = int(manifest["seed"])
    dataset = manifest["dataset"]
    client_num, task_num = p["client_num"], p["task_num"]
    nb_classes = p["nb_classes"]
    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True  # official behavior (non-det.)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    recorder = metrics_recorder.MetricsRecorder(
        out_root=REPO_ROOT / args.output_dir, dataset=dataset,
        method=args.method, seed=seed, run_name=args.run_name,
        meta={
            "manifest": str(Path(args.manifest).resolve()),
            "adapter": "protocol_adapters/run_fl_baseline.py",
            "construction": "constructed FL-only baseline following the "
                            "FedTA paper (arXiv 2412.18355) Table 1 "
                            "setting: full-parameter federated "
                            "fine-tuning of ViT-B/16, fixed nb_classes "
                            "head, sequential task stream, no CL "
                            "mechanism (no replay / prompts / anchors)",
            "local_optimizer": f"Adam(lr={args.lr}, "
                               f"weight_decay={WEIGHT_DECAY})",
            "local_batch": LOCAL_BATCH,
            "local_epoch": args.local_epoch,
            "global_epoch": args.global_epoch,
            "fedprox_mu": args.fedprox_mu if args.method == "fedprox"
                          else None,
            "aggregation": {
                "fedavg": "sample-weighted average of w_i",
                "fedprox": "sample-weighted average of w_i (prox term in "
                           "local loss only)",
                "fednova": "w + a_eff * sum_i (n_i/sum n) * (w_i-w)/a_i; "
                           "a_i = optimizer steps taken, a_eff = "
                           "data-weighted mean a_i",
            }[args.method],
            "eval_pool": "per-(client, task) inner TEST split of the "
                         "manifest, logits masked to the (client, task) "
                         "class_mask (same semantics as "
                         "run_fedmgp._fedmgp_eval_task)",
        })
    ckpt_dir = recorder.run_dir / "checkpoints"
    latest = ckpt_dir / "latest.pth"

    print("=" * 70)
    print(f"[fl-baseline] method      = {args.method}")
    print(f"[fl-baseline] manifest    = {args.manifest}")
    print(f"[fl-baseline] dataset     = {dataset} (nb_classes={nb_classes})")
    print(f"[fl-baseline] clients     = {client_num}, tasks = {task_num}")
    print(f"[fl-baseline] lr={args.lr} local_epoch={args.local_epoch} "
          f"global_epoch={args.global_epoch} batch={LOCAL_BATCH}"
          + (f" mu={args.fedprox_mu}" if args.method == "fedprox" else ""))
    print(f"[fl-baseline] metrics     = {recorder.run_dir}")
    print("=" * 70)

    data = build_client_data(manifest, args.data_path)
    model = build_model(nb_classes, device)

    # resume state: (task, round_in_task) at round boundaries
    start_task, start_round = 0, 0
    if args.resume == "auto":
        if not latest.is_file():
            print("[fl-baseline] --resume auto but no latest.pth; fresh run")
        else:
            st = torch.load(latest, map_location=device)
            model.load_state_dict(st["model"])
            rng_restore(st["rng"])
            start_task, start_round = st["task"], st["round_in_task"]
            print(f"[fl-baseline] resumed at task {start_task} "
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
            client_sds, n_list, a_list = [], [], []
            for cid in range(client_num):
                mu = args.fedprox_mu if args.method == "fedprox" else 0.0
                steps, mean_loss = local_train(
                    model, global_sd, data[cid][task]["train"], args.lr,
                    args.local_epoch, device, args.num_workers, mu=mu)
                client_sds.append({k: v.detach().clone() for k, v in
                                   model.state_dict().items()})
                n_list.append(data[cid][task]["n_train"])
                a_list.append(steps)
                print(f"[fl-baseline] task {task} round {rnd} client {cid} "
                      f"loss {mean_loss:.4f} steps {steps}")
            new_sd = aggregate(args.method, global_sd, client_sds,
                               n_list, a_list)
            model.load_state_dict(new_sd)
            save_resume({"task": task, "round_in_task": rnd + 1,
                         "model": model.state_dict(), "rng": rng_snapshot()},
                        latest)
            print(f"[fl-baseline] task {task} round {rnd} aggregated "
                  f"({time.time() - t0:.0f}s)")

        # task-boundary full-matrix eval: every client x every seen task
        for cid in range(client_num):
            for k in range(task + 1):
                acc = evaluate_task(model, data[cid][k]["test"],
                                    data[cid][k]["class_mask"], nb_classes,
                                    device, args.num_workers)
                recorder.record(cid, task, k, acc, phase="boundary")
                print(f"[fl-baseline] client {cid} after_task {task} "
                      f"test_task {k} acc {acc:.2f}")
        recorder.flush(task_num)
        if args.save_ckpt:
            save_resume({"task": task + 1, "round_in_task": 0,
                         "model": model.state_dict(), "rng": rng_snapshot()},
                        ckpt_dir / f"task_{task:02d}_end.pth")

    print("[fl-baseline] done.")


if __name__ == "__main__":
    main()
