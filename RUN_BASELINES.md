# RUN_BASELINES.md —— 外部基线统一冻结协议运行手册

四个外部基线（FedProTIP / HePCo-Fed(FedCIL) / GLFC / Powder(FedPCR)）均以
**冻结协议 adapter** 接入：官方仓库字节不动，运行时 monkey-patch 只替换
数据划分接口（manifest 注入），输出统一 metrics CSV 树。
设计细节见 `protocol_adapters/PENDING_ADAPTERS.md` 与各 `run_*.py` 头部
docstring；本文件只给命令与自检。

## 0. 前置条件

- **环境**：conda env `FedSMTA`（torch + timm==0.6.7 等），repo 根目录执行。
- **manifest（数据划分真值，两数据集共用）**：
  - CIFAR-100：`protocol_manifests/cifar100_c5t5p15s20_seed42.json`
  - ImageNet-R：`protocol_manifests/inr_c5t5p40s5_seed42.json`
- **数据**（`--data-root` 默认 `local_datasets/`）：
  - CIFAR-100：torchvision 目录布局（`cifar-100-python/`），adapter 以
    `download=False` 直读，需预放好；
  - ImageNet-R：`local_datasets/imagenet-r/{train,test}`（主仓 Imagenet_R
    排序遍历域，四个 adapter 一致）。
- **ViT 权重**（HePCo / Powder 共用）：timm 0.6.7 的
  `vit_base_patch16_224(pretrained=True)` 实际引用 **augreg npz**
  （`B_16-i21k-300ep-...res_224.npz`，约 346MB），缓存于
  `~/.cache/torch/hub/checkpoints/`；离线机器需预先放置。
- **GPU**：单卡即可。实测显存：GLFC ~5.6GB；Powder INR 冒烟
  （batch 16）~3.3GB。autodl 本机注意：GPU 训练不能在 Trae 沙箱内跑
  （CUDA Error 304），须 nohup 脱离启动。

## 1. 输出树（四个 adapter 一致）

```
output/{cifar100|ImageNet-R}/{method}/{run_name}/seed_{seed}/
  ├─ metrics/records.jsonl, accuracy_matrix.csv, summary_metrics.csv,
  │        client_*_accuracy.csv, ...
  ├─ drops.json        # 冻结协议必须 drop 的样本/类记账
  ├─ meta.json
  └─ training_log/     # 官方训练日志原样保留
```

论文口径注记：FedProTIP / HePCo 无回放，轮转公共类已到达样本 drop
（CIFAR 有、INR 零），行注 "uses first-arrival samples only; no replay"；
GLFC / Powder 为 rehearsal/全宽方法，零 drop、合法再训练，行注
"uses replay"（GLFC：exemplar memory + 梯度反演回放）。

## 2. 每方法命令（gate → 冒烟 → 正式跑）

所有命令在 repo 根目录执行。gate 报告落
`protocol_manifests/gate_<method>_<dataset>_report.json`。

### 2.1 FedProTIP（resnet18）

```bash
# gate（两数据集各一次）
python protocol_adapters/gate_fedprotip.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/gate_fedprotip.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json

# 冒烟（约分钟级）
python protocol_adapters/run_fedprotip.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
    --run-name smoke_l1 --limit-tasks 1 --epochs 1 --local-epochs 1

# 正式跑（默认：epochs=5 轮/任务 × local-epochs=30，batch 64）
python protocol_adapters/run_fedprotip.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/run_fedprotip.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json
```

### 2.2 HePCo-Fed / FedCIL（ViT-B/16 + L2P + 生成器 KD）

```bash
python protocol_adapters/gate_hepco.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/gate_hepco.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json

# 冒烟（--local-epochs 必须 ≥2：K-1 做余弦分母；--limit-tasks 必须 2：
#       limit=1 时 hepco "最后任务 schedule 减半"会把训练砍没）
python protocol_adapters/run_hepco.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
    --run-name smoke_l2 --limit-tasks 2 --rounds 1 --local-epochs 2

# 正式跑（默认：rounds=5 轮/任务 × local-epochs=30）
python protocol_adapters/run_hepco.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/run_hepco.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json
```

⚠️ 重跑前先 `pgrep -f run_hepco` 查孤儿进程（hepco 官方 resume 语义会
load 旧模型，曾造成同 run 双份记录）。

### 2.3 GLFC（resnet18_cbam，rehearsal + proxy 梯度反演）

```bash
python protocol_adapters/gate_glfc.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/gate_glfc.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json

# 冒烟
python protocol_adapters/run_glfc.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
    --run-name smoke_l2r1e1 --limit-tasks 2 --rounds 1 --epochs-local 1

# 正式跑（默认：rounds=5 × epochs-local=30，batch 128，memory_size 2000）
python protocol_adapters/run_glfc.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/run_glfc.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json
```

⚠️ **proxy 反演已封章 C'（2026-11-12 实测+验证）**：GLFC 官方隐私监控
（`ProxyServer.reconstruction`）仅在 entropy-signal 轮触发（≈ 首轮 +
每任务边界首轮，~6 事件/跑）。反演是 batch-1 LeNet 双反传，
launch-bound——本机 GPU 上 282ms/closure vs CPU 13ms（21×）。已实施
**C' 设备重路由封章**（反演跑 CPU；数学恒等、零截断、零迭代削减，
训练/蒸馏/聚合仍在 GPU），gate 断言官方锚点 verbatim（B12b/C3c），
meta.json 记 `proxy_recon_device`。C' 冒烟 0.73h 贯通（GPU 原样同负载
~17h）。正式跑 ≈ **~1 天/数据集**（训练段为主 ~16h + proxy ~4h）。
论文 GLFC 行注："proxy monitor inversion on CPU (device placement
only; identical arithmetic)"。

### 2.4 Powder / FedPCR（ViT-B/16 prompt 双知识转移）

```bash
python protocol_adapters/gate_powder.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/gate_powder.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json

# 冒烟（batch 16 留显存余量；正式跑用默认 batch 64）
python protocol_adapters/run_powder.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
    --run-label smoke_l2r1e1 --limit-tasks 2 --rounds 1 --epochs-local 1 --batch-size 16

# 正式跑（默认：rounds=5 × epochs-local=10，batch 64，官方 INR 超参）
python protocol_adapters/run_powder.py --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json
python protocol_adapters/run_powder.py --manifest protocol_manifests/inr_c5t5p40s5_seed42.json
```

## 3. 冒烟自检清单

1. **gate 全 PASS**：8 份报告（4 方法 × 2 数据集）无 FAIL；
2. **冒烟 exit 0**，输出树完整：`records.jsonl` 每任务边界有记录、
   `accuracy_matrix.csv` 的 R 矩阵行数 = 冒烟任务数（limit-tasks 2 → 2 行）、
   `client_*_accuracy.csv` / `drops.json` / `meta.json` 齐全；
3. **drops 口径**：INR 各方法 drops=0；CIFAR 的 FedProTIP/HePCo >0
   （见 drops.json），GLFC/Powder =0；
4. 重跑前 `pgrep -f run_<method>` 清查孤儿进程；
5. 正式跑输出建议 `run_name` 用 `manifest_adapter`（默认）以便区分冒烟。
