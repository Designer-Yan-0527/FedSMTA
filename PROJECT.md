# FedSMTA 项目结构

FedTA：基于 prompt 的联邦持续学习（Federated Continual Learning）项目。
服务器维护全局 prompt / 原型 / 分类头，多个客户端在私有持续任务流上本地训练，
通过 BGPS（原型贪心相似度选择/修复）与 SIKF（知识蒸馏融合 prompt）做聚合。

**当前只支持 `cifar100` 和 `ImageNet-R` 两个数据集**，其他数据集配置与代码已删除。

## 1. 目录总览

```
FedSMTA/
├── main.py                           # 训练入口：参数解析 → 数据划分 → Server_DF 启动
├── runtime_utils.py                  # Phase 0 运行时工程：run 目录 / Tee 日志 /
│                                     #   原子 checkpoint / resume / RNG 状态 / accuracy 读写
├── utils.py                          # 通用工具
├── config/datasets_delay.py          # 全部命令行参数定义（两数据集共用）
│
├── data/                             # 数据集加载与划分
│   ├── cifar100_subset_spliter.py    #   CIFAR-100 划分（公共类 Dirichlet 分配）
│   ├── imagenet_r_subset_spliter.py  #   ImageNet-R 划分（每客户端 40 私有类、0 公共类）
│   ├── continual_datasets.py         #   Imagenet_R 类：tar 解压检查、一次性 80/20 固化
│   └── iCIFAR100c.py                 #   通用包装：surrogate/test 子集、ragged 图片兼容
│
├── Models/                           # 模型与联邦逻辑
│   ├── Server_DF.py                  #   服务器：train_clients 主循环、BGPS、SIKF、
│   │                                 #   fed_avg_head、广播、checkpoint/评估
│   ├── Client_DF.py                  #   客户端：get_data (70/30)、本地训练 (prompt +
│   │                                 #   Tail_Anchor)、A2 post-step restore、评估
│   ├── Tail_Anchor.py                #   Tail Anchor（key + anchor_pool，top-1 路由）
│   ├── classification_head.py        #   per-task Chead
│   ├── global_prompt.py              #   prompt pool
│   ├── vision_transformer*.py        #   ViT 骨干（含 L2P 风格变体）
│   └── models.py                     #   timm 模型注册
│
├── diagnostics/                      # 诊断脚本（全部 observer-only，复刻训练 RNG 流）：
│   │                                 #   extract_features / compatibility_drift (2×2) /
│   │                                 #   analyze_class_retention (4combo) / verify_slot_binding (D0) /
│   │                                 #   collision_exposure (D1 v2) / analyze_landing_spots /
│   │                                 #   verify_tkr / a1_snapshot_eval / ncm_probe (A5 P1)
├── tests/                            # Gate 0A/0B/0C + A2/A1 回归测试
│
├── protocol_manifests/               # 冻结划分真值（cifar100_c5t5p15s20 / inr_c5t5p40s5，
│                                     #   ×seed42；外部基线统一协议的数据源）
├── protocol_adapters/                # 外部基线 adapter：run_fedmgp / run_c2p /
│                                     #   run_fl_baseline (FedAvg/FedProx/FedNova 自实现) /
│                                     #   gate_* / manifest_splitter / metrics_recorder
├── external_baselines/               # 官方仓库克隆（零编辑）：FedMGP、NeurIPS2025-C2Prompt
│
├── pretrain_model/ViT-B_16.npz       # ViT-B/16 预训练权重（original_model 用）
├── RUNCOMMANDS.md                    # 全部运行/诊断/测试命令
├── FEDSMTA_ROADMAP.md                # 研发路线（结论主干 / 失败存档 / A2+A5 活跃方向）
└── output/                           # 训练产物（结构见 RUNCOMMANDS.md §8）
```

## 2. 核心执行流程

```
main.py → 解析参数 → 固定种子 → RunManager → 数据划分 → Server_DF.start()
  每轮：clients[j].update_data()（任务切换时 70/30 切分并记录索引）
       → clients[j].train()（prompt + Tail_Anchor；fedta_a2 下 post-step restore）
       → BGPS 原型聚合/修复 → SIKF kd_fusion_prompt → fed_avg_head → 广播
  任务边界：全任务评估 → 指标 CSV → checkpoint
```

外部基线路径：`protocol_adapters/run_*.py` 读冻结 manifest → monkey-patch 官方仓库
（FedMGP/C²P）或自实现（FL 三基线）→ `MetricsRecorder` 落 RunManager 同格式 CSV。

## 3. 关键架构约定

- **共享 ViT 实例**：所有客户端共享服务器传入的同一 ViT；prompts 以 deepcopy 保存在客户端侧
- **original_model**：冻结预训练 ViT，训练时提供 cls_features 做蒸馏/特征引导
- **method flag**：`--method fedta`（read-only 基线）/ `fedta_a2`（接口保护，`--a2_arm key|anchor|both`）
- **checkpoint 一致性**：`data_split_indices` 记录 70/30 划分；外层 manifest 逐位校验；RNG 状态最后恢复
- **ImageNet-R 划分固化**：首次运行 80/20 移动到 train/test 子目录后复用

## 4. 环境依赖

Python 3.9+ / PyTorch 2.x（开发环境 2.8.0+cu128；服务器 FedSMTA env 为 torch 1.12.1 + timm 0.6.7）/ timm / torchvision / scikit-learn / numpy / matplotlib / tqdm。外部基线另需 pyyaml + transformers + torch_optimizer（C²P）。

## 5. 已测验失败方案存档

失败判决（多模态主线、slot id 绑定、碰撞链、slot hijacking、落点流量、瞬态命中、routing instability、key 位移直接致伤等）的唯一汇总处是 **FEDSMTA_ROADMAP.md §4**，含预注册判据与一句话依据；旧命令在 git 历史。存活结论主干见 Roadmap §3。

## 6. 当前活跃路线

- **A2 接口保护**（已实现；INR seed42 已跑：遗忘压至 2.5，差距集中在基础精度）
- **A5 共享语义接口**（双速架构；P0 manifest 审计已判决 INR 0 公共类；P1 NCM probe 进行中）
- **外部对比实验**（主表 ~14 行：FL 三基线已实现、FedMGP/C²P adapter 就绪、FedL2P/FedDualP/GLFC 等在队列）
- 双机分工：izfree（4090 24GB）跑对比实验（C²P 除外 OOM）；AutoDL（4090 48GB）跑主线
