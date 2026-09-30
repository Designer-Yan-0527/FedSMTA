# FedSMTA 项目结构

FedTA：基于 prompt 的联邦持续学习（Federated Continual Learning）项目。
服务器维护全局 prompt / 原型 / 分类头，多个客户端在私有持续任务流上本地训练，
通过 BGPS（原型贪心相似度选择/修复）与 SIKF（知识蒸馏融合 prompt）做聚合。

**当前只支持 `cifar100` 和 `ImageNet-R` 两个数据集**，其他数据集（5datasets、svhn-mnist、office_home 等）的配置与代码已删除。

## 1. 目录总览

```
FedSMTA/
├── main.py                           # 训练入口：参数解析 → 数据划分 → Server_DF 启动
├── runtime_utils.py                  # Phase 0 运行时工程：run 目录管理 / Tee 日志 /
│                                     #   原子 checkpoint 保存 / resume / RNG 状态 / accuracy 读写
├── utils.py                          # 通用工具：accuracy、build_transform、
│                                     #   global_distillation_loss、CosineSimilarityClassifier
├── test.py                           # 一次性数据脚本（从 imagenet2012 每类采样，非测试文件）
│
├── config/
│   └── cifar100_delay.py            # 全部命令行参数定义（cifar100 / ImageNet-R 共用，
│                                     #   数据集由 --data_name 选择）
│
├── data/                             # 数据集加载与划分
│   ├── cifar100_subset_spliter.py   #   CIFAR-100 划分：客户端×任务，公共类 Dirichlet 分配，
│   │                                 #   CustomedSubset 包装；process_testdata 产出代理/测试集
│   ├── imagenet_r_subset_spliter.py #   ImageNet-R 划分（同上结构，200 类）
│   ├── continual_datasets.py        #   Imagenet_R 数据集类：tar 解压检查、首次 80/20 划分
│   │                                 #   （文件移动到 train/test 子目录，一次性固化）、按类加载
│   └── iCIFAR100c.py                #   通用数据集包装：surrogate/test 子集构建，
│                                     #   兼容 ragged 图片数据（ImageNet-R 尺寸不一）
│
├── Models/                           # 模型与联邦逻辑
│   ├── Server_DF.py                 #   服务器：train_clients 主循环、BGPS
│   │                                 #   (choose_best_proto_greedy_similarity_fixed_key)、
│   │                                 #   SIKF (kd_fusion_prompt)、fed_avg_head、广播、
│   │                                 #   Phase 0 checkpoint/评估/指标
│   ├── Client_DF.py                 #   客户端：get_data (70/30 划分并记录索引)、本地训练
│   │                                 #   (prompt + Tail_Anchor)、原型提取、评估、
│   │                                 #   state_dict_for_checkpoint / rebuild_data_from_indices
│   ├── Tail_Anchor.py               #   Tail Anchor 模块（key + anchor_pool）
│   ├── classification_head.py       #   Chead 分类头（逐任务使用）
│   ├── global_prompt.py             #   Global_Prompt（prompt pool）
│   ├── vision_transformer.py        #   ViT 骨干网络
│   ├── vision_transformer__l2p.py   #   L2P 风格 ViT（带 prompt 支持，训练用）
│   ├── models.py                    #   timm 模型注册
│   └── CosineSimilarityClassifier.py
│
├── pretrain_model/
│   └── ViT-B_16.npz                 # ViT-B/16 预训练权重（服务器端 original_model 使用）
│
├── RUNCOMMANDS.md                    # 运行命令说明
└── output/                           # 训练产物（运行时自动生成，见 RUNCOMMANDS.md 第 8 节）
```

## 2. 核心执行流程

```
main.py
  ├─ 解析参数（config/cifar100_delay.py）
  ├─ 固定种子（torch / numpy / random）
  ├─ RunManager(args)                 # 建 run 目录、Tee 日志、解析 resume 目标
  ├─ 数据划分（cifar100 / ImageNet-R spliter）
  │    → client_data[i][task]，client_mask[i][task]（每个客户端的任务数据与类别）
  ├─ original_model = ViT-B/16 预训练模型（冻结，提供 cls_features）
  └─ Server_DF(...).start()
       ├─ init_client()               # 创建 client_num 个 Client_DF
       ├─ load_checkpoint()           # resume 时恢复状态（RNG 状态最后恢复）
       └─ train_clients()             # 每轮：
            1. clients[j].update_data()   # 切换任务时做 70/30 划分（记录索引）
            2. clients[j].train()         # 本地训练 prompt + Tail_Anchor
            3. BGPS 原型聚合/修复          # threshold 由 --threshold 控制
            4. SIKF kd_fusion_prompt      # 任务最后一轮之外的轮次执行
            5. fed_avg_head               # 分类头联邦平均
            6. 广播全局原型/头/prompt
            7. (任务结束) 全任务评估 → 指标 → checkpoint → 日志落盘
```

## 3. 关键架构约定

- **共享 ViT 实例**：所有客户端共享服务器传入的同一个 ViT（`self.model`）；各客户端的 prompts 以 deepcopy 保存在客户端侧，训练前 `load_prompts` 加载。
- **original_model**：服务器加载的冻结预训练 ViT（`pretrain_model/ViT-B_16.npz`），训练时提供 `cls_features` 做蒸馏/特征引导，不参与更新。
- **分类头**：`client.heads[task]` 存放每个任务的 `Chead` 模块；服务器侧有 `global_head` 与 `fix_keys`（BGPS 修复的类别）。
- **checkpoint 一致性**：`data_split_indices` 记录每个任务的 70/30 划分索引，resume 时精确重建 test_loader 与 traindata；RNG 状态（python/numpy/torch/cuda）在所有状态恢复完成后**最后**恢复。
- **ImageNet-R 划分固化**：首次运行时把图片移动到 `train/`、`test/` 子目录，此后直接复用磁盘上的划分结果，保证多次运行一致。

## 4. 环境依赖

- Python 3.9+
- PyTorch 2.x（开发环境为 2.8.0+cu128）
- timm（模型创建 / 优化器 / 调度器）
- torchvision、scikit-learn、numpy、matplotlib、tqdm
- 无需 deeplake（该依赖原本只用于已删除的 office_home 数据集）
