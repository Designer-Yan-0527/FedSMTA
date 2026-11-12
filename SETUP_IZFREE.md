# SETUP_IZFREE —— izfree 机器全实验部署手册

> 目标：在 izfree 上跑完 A2+L+B 方案的全部论文实验 = 主方法四臂矩阵 × 2 数据集 + 4 个外部基线 × 2 数据集（+ 可选内部基线）。
> 权威命令来源：主方法见 `RUN.md`，外部基线见 `RUN_BASELINES.md`；本文件只负责**环境搭建、资产放置、执行顺序与自检**，不重复命令正文。

## 0. 前置要求

| 项 | 要求 |
|---|---|
| GPU | 单卡即可（实验全部单卡），显存 ≥ 12GB 稳妥（主方法 ViT-B/16 训练 ~10GB 内） |
| 磁盘 | 预留 ≥ 100GB：数据 ~35GB（ImageNet-R ~7GB 解压 ×2 份布局 + CIFAR ~150MB）+ checkpoint 约 350MB/轮 × 多个 run |
| CUDA driver | ≥ CUDA 11.3 runtime 兼容（驱动 ≥ 470.x） |
| 网络 | izfree 若离线，见 §3/§4/§5 的离线预放步骤 |

## 1. 拉代码

```bash
git clone git@github.com:Designer-Yan-0527/FedSMTA.git
cd FedSMTA
git log --oneline -1   # 记下 commit，论文实验须统一版本
```

## 2. conda 环境

与 autodl 验证机完全一致的版本（已跑通全部 gate/冒烟）：

```bash
conda create -n FedSMTA python=3.10 -y
conda activate FedSMTA
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.6.7 numpy==1.23.5 scipy scikit-learn pandas tqdm \
    Pillow==9.2.0 matplotlib==3.5.3
```

> 离线机器：在联网机器上 `pip download` 同版本 whl 打包拷入，再 `pip install --no-index --find-links .`。

## 3. 数据集放置（本地导入，代码不联网下载）

```
local_datasets/
├── cifar-100-python/          # torchvision 标准布局（train/test bin + meta）
└── imagenet-r/                 # 主仓与 adapter 共用
    ├── train/                 # 200 个类别子目录（或放 imagenet-r.tar 由主仓首跑自动划分）
    └── test/
```

- **CIFAR-100**：`local_datasets/cifar-100-python/` 必须预放好，缺失直接报错（`download=False` 直读）。
- **ImageNet-R**：主仓 `ImagenetR_spliter` 接受 `imagenet-r/`（200 类目录平铺）或 `imagenet-r.tar`；首次运行做一次性 80/20 划分（移动到 train/test 子目录后复用）。四个外部基线 adapter 读 `local_datasets/imagenet-r/{train,test}` 排序域——**直接放划分好的 train/test 即两边通用**。
- 数据划分真值 = `protocol_manifests/{cifar100_c5t5p15s20_seed42.json, inr_c5t5p40s5_seed42.json}`（随 repo 入库，无需生成）。

## 4. ViT 权重预放（离线机器必须）

用户自行下载，两个文件：

| 权重 | 放置路径 | 用途 | 大小 |
|---|---|---|---|
| `ViT-B_16.npz`（L2P 系 ImageNet-21k 预训练 ViT-B/16，即 L2P 官方仓 `pretrain_model/` 同名文件） | `FedSMTA/pretrain_model/ViT-B_16.npz` | 主方法 main.py（`pretrained_cfg['file']` 硬引用） | ~394MB |
| `B_16-i21k-300ep-...res_224.npz`（timm 0.6.7 `vit_base_patch16_224(pretrained=True)` 实际引用的 augreg 权重） | `~/.cache/torch/hub/checkpoints/` | HePCo / Powder adapter | ~346MB |

> .gitignore 已排除两个 npz（超 GitHub 100MB 单文件限制）。

## 5. 主方法四臂矩阵（A2+L+B）

命令正文照抄 `RUN.md`（四臂：base / `--a2_arm key` / `--dsar` / `--a2_arm key --dsar`，CIFAR 与 INR 各四条，DSAR 配方由 `--data_name` 自动分派）。单卡串行队列示例：

```bash
# 每条命令在 repo 根目录执行；建议逐 dataset 排队
nohup bash -c '
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 --global_epoch 5 --local_epoch 30 \
    --batch-size 16 --lr 0.001 --seed 42 --run_name cifar_base_p15
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 --global_epoch 5 --local_epoch 30 \
    --batch-size 16 --lr 0.001 --seed 42 --a2_arm key --run_name cifar_a2_key_v2
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 --global_epoch 5 --local_epoch 30 \
    --batch-size 16 --lr 0.001 --seed 42 --dsar --run_name cifar_b1_dsar_on
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 --global_epoch 5 --local_epoch 30 \
    --batch-size 16 --lr 0.001 --seed 42 --a2_arm key --dsar --run_name cifar_a2_dsar_on
' > queue_cifar.log 2>&1 &
```

INR 四条同理（加 `--surrogate_num 5 --private_class_num 40`，命令见 RUN.md）。

发车自检（每条启动后，照 RUN.md「发车自检」）：
1. `output/cifar100/fedta/<run>/seed_42/args.json` 旗标与臂一致；
2. 进任务 1 后 train.log 出现 `[A2] client X task 1 arm=key: protected ...`；
3. DSAR 臂广播边界后 `b_status`/`l_status` 出现 `FITTED`；
4. INR DSAR 臂 sidecar meta `recipe: "dsar-inr-v1"`（CIFAR 为 `"dsar-b1-cifar"`）。

磁盘紧张加 `--save_every 5 --keep_last 2`；断点续训 `--run_name <原run> --resume auto`；不要叠加 `--bsab_pair_capture` 等插桩开关。

## 6. 外部基线（FedProTIP / HePCo / GLFC / Powder）

完整 gate → 冒烟 → 正式跑命令照抄 `RUN_BASELINES.md`（manifest 注入、输出树、drops 口径、论文行注全在该文档）。要点提醒：

1. **先跑 8 个 gate**（4 方法 × 2 manifest，CPU 级、分钟级），全 PASS 再继续；
2. 每方法先冒烟再正式跑（命令在 RUN_BASELINES.md §2）；
3. **重跑前 `pgrep -f run_<method>` 清孤儿进程**（hepco 官方 resume 语义曾造成双份记录）；
4. GLFC 的 proxy 反演 C' 封章（CPU 重路由）、`num_workers=0` 封章已内置在 adapter，无需配置；
5. adapter 启动自带 numpy SVD 自检（autodl 曾发现 LAPACK 损坏；izfree 未测，自检会自动判定并在坏时引流 scipy——若日志出现 `svd_robust` 提示，说明该机 numpy 也有问题，数值不受影响，但建议记录）；
6. 显存参考：FedProTIP ~4GB；GLFC ~5.6GB（+CPU 反演）；Powder INR 冒烟(batch16) ~3.3GB、正式 batch 64 留余量；HePCo ViT 同主方法量级。

## 7. 可选内部基线（如论文需要）

- FL-only 构造基线（FedAvg / FedProx / FedNova）：`protocol_adapters/run_fl_baseline.py`，命令档案见 RUNCOMMANDS.md §19；
- Prompt-only 构造基线：`protocol_adapters/run_prompt_baseline.py`；
- FedMGP：`protocol_adapters/run_fedmgp.py`（gate：`gate_fedmgp.py`，INR 用 `gate_fedmgp_inr.py`）。

## 8. 发车前冒烟（主方法，分钟级）

```bash
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 --global_epoch 5 --local_epoch 1 \
    --batch-size 16 --lr 0.001 --seed 42 --max_rounds 1 --run_name izfree_smoke
```

自检：exit 0；`output/cifar100/fedta/izfree_smoke/seed_42/` 下 args.json / logs/train.log / checkpoints/latest.pth / metrics/ 齐全；train.log 无 traceback。

## 9. 产物位置与汇总

```
output/{cifar100|ImageNet-R}/{method}/{run_name}/seed_{seed}/
    ├── args.json / command.txt / git_commit.txt
    ├── logs/train.log, checkpoints/, metrics/accuracy_matrix.csv,
    │   summary_metrics.csv (AA/Forgetting/BWT), train_log.csv, ...
外部基线另有 drops.json / meta.json / records.jsonl / training_log/
```

四臂矩阵主表读各 run 的 `summary_metrics.csv` + `accuracy_matrix.csv`（AA / Forgetting / BWT / R[t][k]）。
