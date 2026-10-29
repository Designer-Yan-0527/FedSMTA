"""Refit input-path ablation probe (A5-v2 §23.6-1, INR M2 diagnosis).

背景：INR T0 上 M2 refit 落后官方（cifar 是 +0.93 GREEN，INR round 0
−0.65 且 round 1 差距扩大到 −1~−2），两个预注册嫌疑需要分辨：

  嫌疑 1（配方过拟合）：refit 固定配方（hidden 512 / dropout 0.1 /
    100 epochs）是在 cifar ~1.7k 行上调的；INR 每 client 只有
    ~0.9-1.2k 行，MLP 可能过拟合增强特征、提前饱和。
  嫌疑 2（x1 路径天花板）：探针 +4.04 headroom 是 768 维 pre-anchor
    z_op 上测的；refit 吃的是 anchor-concat + top-1 硬路由的
    1536 维 x1，早期路由选错 anchor 时 x1 = 特征 + 噪声锚。

分辨方法：同一 checkpoint、同一批样本，提取 z_op 与 x1 两套特征，
各用多档配方训探针头、corrected（eval 模式、dropout 关）评测：

  x1_refit   : x1 + refit 原配方（faithful，不标准化）
  z_refit    : z_op + 同配方
  x1_light   : x1 + 轻配方（hidden 256 / dropout 0.3 / 30 epochs）
  z_light    : z_op + 同轻配方
  x1_probe256: x1 + 探针原配方（hidden 256 / dropout 0.1 / 100 ep /
                train-only 标准化）—— 探针数字在 x1 上的移植
  z_probe256 : z_op + 同上 —— 应复现 §22 的 +4.04 headroom（sanity）

内部效度锚点（不训练，直接从 checkpoint 评测）：
  native     : checkpoint 里的联训头（model.head，hook 已还原），
               eval 模式 + mask 限制 —— F2 归因的正确参照系
  saved_refit: checkpoint 里存的 refit 头（heads[t]，M2 臂 ON 时即
               hook 替换后的头）—— 与日志里 [A5-v2] 行的差应只
               剩 eval 端 dropout（refit 头 p=0.1）

判读表（预注册，client 均值口径）：
  z_probe256 - native ≈ +4                → 提针复现，管线可信
  saved_refit - native ≈ 日志中 refit 增益  → hook 复现，管线可信
  z_refit - x1_refit >= 1pp               → 嫌疑 2（x1 路径）主导
  x1_light - x1_refit >= 1pp              → 嫌疑 1（配方过拟合）主导
  两者都不动                              → 双嫌疑皆弱，问题在别处
                                             （增强噪声 / 特征太生）

用法（repo 根目录）：
  python diagnostics/probe_refit_input_ablation.py \
      --run_dir output/ImageNet-R/fedta/inr_a5v2_M2_refit/seed_42 \
      --checkpoints round_0000.pth,round_0001.pth \
      --out_csv diagnostics_output/refit_input_inr_m2.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diag_utils import bootstrap_server, resolve_checkpoint  # noqa: E402
from Models.Client_DF import a5_stable_seed  # noqa: E402

# refit 原配方（与 Models/Client_DF.py 的 A5_* 常量逐项一致）
REFIT_HIDDEN = 512
REFIT_DROPOUT = 0.1
REFIT_EPOCHS = 100
REFIT_LR = 1e-3
REFIT_WD = 1e-4
REFIT_BATCH = 64

RECIPES = [
    # (名称, hidden, dropout, epochs, lr, wd, standardize)
    ("x1_refit",    REFIT_HIDDEN, REFIT_DROPOUT, REFIT_EPOCHS, REFIT_LR, REFIT_WD, False),
    ("z_refit",     REFIT_HIDDEN, REFIT_DROPOUT, REFIT_EPOCHS, REFIT_LR, REFIT_WD, False),
    ("x1_light",    256, 0.3, 30, REFIT_LR, REFIT_WD, False),
    ("z_light",     256, 0.3, 30, REFIT_LR, REFIT_WD, False),
    ("x1_probe256", 256, 0.1, REFIT_EPOCHS, REFIT_LR, REFIT_WD, True),
    ("z_probe256",  256, 0.1, REFIT_EPOCHS, REFIT_LR, REFIT_WD, True),
]
# 配方 -> 用哪套特征
FEAT_OF = {"x1_refit": "x1", "x1_light": "x1", "x1_probe256": "x1",
           "z_refit": "z_op", "z_light": "z_op", "z_probe256": "z_op"}


def parse_args():
    p = argparse.ArgumentParser("refit input-path ablation probe")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoints", default="round_0000.pth,round_0001.pth",
                   help="逗号分隔的 checkpoint（相对 run_dir）")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None,
                   help="默认取 run args.json 里的 device")
    p.add_argument("--batch_size", default=16, type=int,
                   help="特征提取 batch（与 hook 收集一致：16）")
    p.add_argument("--seed_base", default=20261029, type=int,
                   help="非 faithful 配方的探针种子基")
    return p.parse_args()


def fit_head_opt(z_tr, y_tr, classes, *, seed, hidden, epochs, lr, wd,
                 dropout, batch_size, standardize, device):
    """probe_oracle_headroom.fit_head 的可关标准化版。

    masked-CE 等价语义：只给 allowed 类建输出单元（softmax 定义域
    相同，梯度在 allowed 列上与 200 列 + (-inf) mask 完全一致）。"""
    z_tr = np.asarray(z_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    if standardize:
        mu = z_tr.mean(axis=0)
        sd = z_tr.std(axis=0) + 1e-8
    else:
        mu = np.zeros(z_tr.shape[1], dtype=np.float32)
        sd = np.ones(z_tr.shape[1], dtype=np.float32)
    z = (z_tr - mu) / sd
    n, d = z.shape
    remap = {int(c): i for i, c in enumerate(classes)}
    yt = torch.tensor([remap[int(c)] for c in y_tr], dtype=torch.long,
                      device=device)
    x = torch.tensor(z, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    c = len(classes)
    model = torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.ReLU(),
        torch.nn.Dropout(dropout), torch.nn.Linear(hidden, c))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    bs = min(batch_size, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[idx]),
                                                     yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd, remap


def masked_predict(model, mu, sd, remap, z_te, allowed, device):
    """eval 模式（dropout 关）+ 类别 mask 限制的 argmax——corrected
    口径 + R[t][k] 语义。"""
    zt = torch.tensor((np.asarray(z_te, dtype=np.float32) - mu) / sd,
                      dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(zt).cpu().numpy()
    cols = [remap[int(c)] for c in allowed]
    best = logits[:, cols].argmax(axis=1)
    return np.asarray([allowed[i] for i in best], dtype=np.int64)


def acc(pred, gold):
    return 100.0 * float(np.mean([p == int(g) for p, g in zip(pred, gold)]))


def extract_client_task(client, server, t, device, batch_size, num_workers):
    """train/test 两 split 的 z_op（768 维 pre-anchor）与 x1（1536 维
    anchor-concat），native evaluate 语义（task_id=client.task_id，
    vit train=True —— ViT drop=0 前向确定）。"""
    feats = {"train": {"z_op": [], "x1": [], "y": []},
             "test": {"z_op": [], "x1": [], "y": []}}
    for split in ("train", "test"):
        indices = list(client.data_split_indices[t][split])
        if not indices:
            raise RuntimeError(
                f"client {client.id} task {t} split {split} 为空")
        subset = Subset(client.train_data[t], indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        with torch.no_grad():
            for inp, tgt in loader:
                inp = inp.to(device, non_blocking=True)
                out_ref = server.origin_model(inp)
                out_op = client.vit(inp, task_id=client.task_id,
                                    cls_features=out_ref["pre_logits"],
                                    train=True)
                z_op = out_op["feat"].float()
                _, x1, _ = client.model(z_op, tgt.to(device))
                feats[split]["z_op"].append(
                    z_op.float().cpu().numpy())
                feats[split]["x1"].append(
                    x1.float().cpu().numpy())
                feats[split]["y"].append(
                    tgt.cpu().numpy().astype(np.int64))
    return {s: {k: np.concatenate(v[k]) for k in v}
            for s, v in feats.items()}


def main():
    args_cli = parse_args()
    run_dir = Path(args_cli.run_dir)
    out_csv = Path(args_cli.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ckpts = [c.strip() for c in args_cli.checkpoints.split(",") if c.strip()]

    rows = []
    summaries = []
    for ckpt in ckpts:
        ckpt_path = resolve_checkpoint(run_dir, ckpt)
        run_args, server = bootstrap_server(
            run_dir, checkpoint=ckpt_path,
            data_path=args_cli.data_path, device=args_cli.device)
        device = torch.device(run_args.device)
        completed_round = int(getattr(server, "start_round", 0)) - 1
        server.origin_model.to(device).eval()
        server.model.to(device)

        for client in server.clients:
            if client.task_id < 0:
                continue
            client.vit.to(device)
            client.model.to(device).eval()
            if client.prompts is not None:
                client.vit.load_prompts(client.prompts)
            for t in range(client.task_id + 1):
                if t not in client.data_split_indices:
                    continue
                data = extract_client_task(
                    client, server, t, device, args_cli.batch_size,
                    run_args.num_workers)
                allowed = sorted(int(c) for c in client.class_mask[t])
                y_te = data["test"]["y"]
                z_te, x_te = data["test"]["z_op"], data["test"]["x1"]

                row = {"checkpoint": Path(ckpt_path).name,
                       "round": completed_round, "client": client.id,
                       "task": t,
                       "n_train": len(data["train"]["y"]),
                       "n_test": len(y_te)}

                # 内部效度锚点 1：原生联训头 corrected（F2 归因参照系）
                joint = client.model.get_head()
                joint.eval()
                with torch.no_grad():
                    zt = torch.tensor(x_te, dtype=torch.float32,
                                      device=device)
                    logits = joint(zt).cpu().numpy()
                jmap = {int(c): i for i, c in enumerate(allowed)}
                cols = [jmap[int(c)] for c in allowed]
                # 联训头是 200 输出：mask 列限制
                row["native"] = round(acc(
                    [allowed[i] for i in logits[:, cols].argmax(1)], y_te), 2)

                # 内部效度锚点 2：checkpoint 里存的 refit 头 corrected
                # （与日志 [A5-v2] 行的差应只剩 eval 端 dropout）
                # rebuild_client_head 重建的头在 CPU，需搬到 device
                saved = client.heads[t].to(device)
                saved.eval()
                with torch.no_grad():
                    logits = saved(zt).cpu().numpy()
                row["saved_refit"] = round(acc(
                    [allowed[i] for i in logits[:, cols].argmax(1)], y_te), 2)

                # 六档配方：x1 / z_op × 配方
                for name, hidden, drop, eps, lr, wd, std in RECIPES:
                    z_tr = data["train"][FEAT_OF[name]]
                    y_tr = data["train"]["y"]
                    zt_eval = x_te if FEAT_OF[name] == "x1" else z_te
                    if name in ("x1_refit", "z_refit"):
                        # faithful 复刻 hook 的种子（不做逐位主张：
                        # 输出列数与 Chead 不同，RNG draw 数不同，
                        # 但配方/数据/种子派生与 hook 完全一致）
                        seed = a5_stable_seed(run_args.seed, "a5refit",
                                              client.id, t,
                                              completed_round)
                    else:
                        seed = a5_stable_seed(args_cli.seed_base, name,
                                              client.id, t, completed_round)
                    model, mu, sd, remap = fit_head_opt(
                        z_tr, y_tr, allowed, seed=seed, hidden=hidden,
                        epochs=eps, lr=lr, wd=wd, dropout=drop,
                        batch_size=REFIT_BATCH, standardize=std,
                        device=device)
                    row[name] = round(acc(
                        masked_predict(model, mu, sd, remap, zt_eval,
                                       allowed, device), y_te), 2)
                rows.append(row)

        # 本 checkpoint 的 client 均值摘要
        sub = [r for r in rows
               if r["checkpoint"] == Path(ckpt_path).name]
        keys = ["native", "saved_refit"] + [r[0] for r in RECIPES]
        mean = {k: round(float(np.mean([r[k] for r in sub])), 2)
                for k in keys}
        mean["checkpoint"] = Path(ckpt_path).name
        mean["round"] = completed_round
        summaries.append(mean)
        print(f"\n=== {Path(ckpt_path).name} (round {completed_round}) "
              f"client 均值 ===")
        hdr = f"{'native':>8} {'saved_rf':>9} {'x1_rf':>7} {'z_rf':>7} " \
              f"{'x1_lt':>7} {'z_lt':>7} {'x1_pb':>7} {'z_pb':>7}"
        print(hdr + f"   (native={mean['native']})")
        print(f"{mean['native']:>8.2f} {mean['saved_refit']:>9.2f} "
              f"{mean['x1_refit']:>7.2f} {mean['z_refit']:>7.2f} "
              f"{mean['x1_light']:>7.2f} {mean['z_light']:>7.2f} "
              f"{mean['x1_probe256']:>7.2f} {mean['z_probe256']:>7.2f}")
        print(f"  Δ refit(x1)−native   = "
              f"{mean['x1_refit'] - mean['native']:+.2f}")
        print(f"  Δ 路径 z_refit−x1_refit = "
              f"{mean['z_refit'] - mean['x1_refit']:+.2f}"
              f"   (>=+1pp → 嫌疑2 x1 路径)")
        print(f"  Δ 配方 x1_light−x1_refit = "
              f"{mean['x1_light'] - mean['x1_refit']:+.2f}"
              f"   (>=+1pp → 嫌疑1 过拟合)")
        print(f"  Δ 探针 z_probe256−native = "
              f"{mean['z_probe256'] - mean['native']:+.2f}"
              f"   (应接近 §22 的 +4.04)")
        print(f"  Δ 移植 x1_probe256−native = "
              f"{mean['x1_probe256'] - mean['native']:+.2f}"
              f"   (探针配方在 x1 上的 headroom)")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[refit-input-probe] per-client rows -> {out_csv}")
    print("[refit-input-probe] summary:")
    for s in summaries:
        print("  " + ", ".join(f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
