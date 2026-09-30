# FedTA 运行命令

本项目只支持 `cifar100` 和 `ImageNet-R` 两个数据集，通过 `--data_name` 选择。
所有参数定义在 `config/datasets_delay.py`，两个数据集共用同一套参数。

## 1. 基本调用格式

```bash
python main.py datasets_delay --data_name {cifar100|ImageNet-R} [其他参数]
```

> `datasets_delay` 是子命令名（必带），与数据集无关；数据集由 `--data_name` 决定。

## 2. 运行 CIFAR-100

```bash
python main.py datasets_delay \
    --data_name cifar100 \
    --data_path ./local_datasets \
    --client_num 5 \
    --task_num 5 \
    --private_class_num 15 \
    --global_epoch 5 \
    --local_epoch 30 \
    --batch_size 16 \
    --lr 0.001 \
    --seed 42
```

数据集准备：**本地导入，不联网下载**。`--data_path` 目录下需已存在 `cifar-100-python/`（解压后的 CIFAR-100）；数据缺失时直接报错，不会自动下载。

## 3. 运行 ImageNet-R

```bash
python main.py datasets_delay \
    --data_name ImageNet-R \
    --data_path ./local_datasets \
    --surrogate_num 5 \
    --client_num 5 \
    --task_num 5 \
    --private_class_num 15 \
    --global_epoch 5 \
    --local_epoch 30 \
    --batch_size 16 \
    --lr 0.001 \
    --seed 42
```

数据集准备（**本地导入，不联网下载**）：

1. `--data_path` 目录下需已存在 `imagenet-r/`（解压后的 200 个类别目录）或 `imagenet-r.tar`（存在 tar 且未解压时会本地解压）；数据缺失时直接报错，不会自动下载；
2. 首次运行会做**一次性**的 80/20 随机划分（把图片移动到 `imagenet-r/train/`、`imagenet-r/test/` 子目录，之后直接复用，不会重复划分）；
3. ImageNet-R 共 30000 张图片、200 类，`args.nb_classes` 自动设为 200。

> **注意**：`--surrogate_num` 现在真正生效（旧代码硬编码为 5）。如果想保持与旧代码完全一致的行为，请显式传 `--surrogate_num 5`；默认值为 20。

## 4. 常用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_name` | `cifar100` | 数据集，可选 `cifar100` / `ImageNet-R`（区分大小写） |
| `--data_path` | `./local_datasets` | 数据集根目录 |
| `--output_dir` | `output/` | 训练产物根目录 |
| `--client_num` | 5 | 客户端数量 |
| `--task_num` | 5 | 任务（阶段）数量 |
| `--private_class_num` | 15 | 每个客户端的私有类别数 |
| `--global_epoch` | 5 | 每个任务的联邦通信轮数 |
| `--local_epoch` | 30 | 每轮本地训练 epoch 数 |
| `--batch_size` | 16 | 本地训练 batch size（现已生效，旧代码硬编码 16） |
| `--lr` | 0.001 | 学习率 |
| `--threshold` | 0.25 | BGPS 原型修复阈值（现已生效，旧代码硬编码 0.25） |
| `--surrogate_num` | 20 | 每类代理样本数（ImageNet-R 旧代码硬编码 5） |
| `--seed` | 42 | 随机种子 |
| `--num_workers` | 2 | DataLoader 进程数 |
| `--device` | `cuda` | 训练设备 |

## 5. 运行管理参数（Phase 0 工程，不影响算法）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--run_name` | `''`（自动生成） | 实验名。留空自动生成 `{dataset}_{method}_c.._t.._ge.._le.._lr.._seed.._{hash}` |
| `--resume` | `''` | `''` = 全新运行；`auto` = 从 `RUN_DIR/checkpoints/latest.pth` 恢复；或传 checkpoint 显式路径 |
| `--save_every` | 1 | 每 N 个全局轮保存一次 `round_XXXX.pth` |
| `--keep_last` | 0 | 只保留最近 N 个 `round_XXXX.pth`（0 = 全部保留）。`latest.pth` 和 `task_XX_end.pth` 始终保留 |
| `--eval_all_every` | 0 | 每 N 轮做一次全任务评估（0 = 仅在任务结束时评估） |

## 6. 断点续训（resume）

```bash
# 方式一：自动从最近 checkpoint 恢复（要求 --run_name 等参数与原运行一致）
python main.py datasets_delay --data_name cifar100 --run_name <原run_name> --resume auto

# 方式二：显式指定 checkpoint 路径（run_dir 自动从路径推断）
python main.py datasets_delay --data_name cifar100 \
    --resume output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_02_end.pth
```

说明：

- checkpoint 在**轮边界**保存，恢复时从下一轮继续，训练循环、数据划分、RNG 状态均精确还原；
- 恢复时校验关键参数（数据集、模型、client/task 数、lr、seed 等），不一致会拒绝恢复；
- 旧日志以追加方式保留，accuracy matrix 会接续完整历史。

## 7. 磁盘占用提醒

checkpoint 包含完整 ViT state_dict，**约 350MB+/轮**。默认每轮都保存（`--save_every 1`），一个 25 轮的运行约 9GB。磁盘紧张时建议：

```bash
--save_every 5 --keep_last 2   # 每 5 轮存一次，只保留最近 2 个 round checkpoint
```

## 8. 运行产物目录结构

```
output/
└── {data_name}/                  # cifar100 / ImageNet-R
    └── {method}/                 # 默认 fedta
        └── {run_name}/
            └── seed_{seed}/
                ├── command.txt           # 完整启动命令（追加式）
                ├── args.json             # 全部参数快照（仅首次运行写入）
                ├── git_commit.txt
                ├── logs/
                │   └── train.log         # 终端输出的完整副本（追加式）
                ├── checkpoints/
                │   ├── latest.pth        # 每轮更新，resume 默认入口
                │   ├── round_0001.pth    # 按 --save_every 保存
                │   └── task_00_end.pth   # 每个任务结束时的快照
                └── metrics/
                    ├── accuracy_matrix.csv    # 全任务准确率矩阵
                    ├── client_0_accuracy.csv  # 各客户端准确率
                    └── round_metrics.jsonl    # 每轮指标流水
```
