# 外部基线统一协议 adapter —— 待实现设计 brief（2026-11-11）

目标：给 4 个新选外部方法（FedProTIP / hepco-fed(FedCIL) / Powder(FedPCR) / GLFC）
写 manifest 统一协议 runner + gate，照 FedMGP / C2P 模板。

## 冻结纪律（照抄 run_c2p.py / manifest_splitter.py 头部声明）
- 官方仓库字节不动（external_baselines/ gitignored），全部 runtime monkey-patch
- 只替换数据划分接口；方法自身超参/损失/优化器 = method-native
- bugfix 必须封章：patch 前断言官方源仍含 bad line（防止 repo 更新后静默失效）
- 输出走 protocol_adapters/metrics_recorder.py（RunManager 同格式 CSV 树）
- gate：PartA 原生等价 / PartB 适配器注入 / PartC bugfix 封章，报告落
  protocol_manifests/gate_<method>_report.json

## 【2026-11-11 环境级发现：autodl 的 numpy LAPACK 是坏的】
- 现象：np.linalg.svd 对 (1152,512)/(4608,512) float32 矩阵返回
  **错误奇异值且不报错**（np=8349 vs torch 真值 90；eigvalsh(aᵀa) 给
  31.7 vs 真值 39.34）。偶尔直接抛 "LAPACK: DLASCL parameter 4
  illegal value"。
- 已用三条独立路径定真值：Frobenius 上界、无 BLAS 幂迭代、torch CPU
  (MKL)——numpy OpenBLAS ILP64 多线程/单线程都错；scipy.linalg 幸存。
- **对策（封章式，run_fedprotip.py 的 install_robust_svd()）**：启动自检
  （三种 shape 与 torch.linalg.svdvals 交叉验证），坏则进程内把
  np.linalg.svd 引流到 scipy.linalg.svd；官方文件不动；gate C5 断言官方
  np.linalg.svd 锚点仍在（client_fedprotip.py 6 处）。
- **后续所有 adapter（GLFC 梯度反演几乎必用 LAPACK）都要复用这个自检**；
  izfree 机器未测，自检可移植自动判定。
- 运行须知：GPU 训练必须 dangerouslyDisableSandbox（沙箱挡 CUDA 初始化
  Error 304），且非沙箱 shell 不带 conda env，须用绝对路径
  /root/autodl-tmp/users/ych/.conda/envs/FedSMTA/bin/python。

## 冻结 manifest 语义（已验证，勿再猜）——【2026-11-11 重大修正】
- CIFAR-100 (c5t5p15s20)：5 客户端 × 5 任务，**每客户端每任务 8 类**
  （3 私有 + 5 公共；私有 15/客户端合计，公共 25 类全客户端共享），
  outer[c][t] = {raw_indices(全量 trainset 枚举域), class_mask}，
  inner[c][t] = {train(70%), test(30%)}（客户端训练/评估集）
- **修正：公共类按 (客户端,任务) 轮转**！全局任务 t 的类并集 = 30~33 类
  （非 20），跨任务并集有重叠（156 次出现 > 100 类）：
  - 75 私有类：各只出现 1 次（1 客户端 1 任务）
  - 25 公共类：各出现 5 次（每客户端 1 次），分散在 2~4 个全局任务
- **全局 class-IL 模型的正确调和 = 类按"首次到达"排序**（不是并集直切）：
  新类到达序列 CIFAR = [32, 26, 14, 16, 12]（合 100）；INR = [40×5]
  （INR 全私有，无轮转，每 (c,t) mask 的类都是 t 时刻新类，0 drop）
- **轮转公共类的已到达样本必须 drop**（全局类增量方法无回放机制时，
  例如 FedProTIP _finetune 的 fake_targets=labels-known 会越界崩溃）：
  CIFAR 共 drop 5986 个 inner.train 样本（约占 8.5%），落
  {run_dir}/drops.json + gate 报告，论文该行加注 "uses first-arrival
  samples only; no replay"
- surrogate['indices'] = 服务端代理数据；test_pool['indices'] = 评估池
  （C2P 先例：客户端训练用 inner.train 位置；评估池 = test_pool）

## 各方法注入点（Explore 已探明）
### FedProTIP（external_baselines/FedProTIP）——✅ 已完成（2026-11-11）
- run_fedprotip.py + gate_fedprotip.py，两 manifest gate 全 PASS
  （报告 gate_fedprotip_{cifar100,imagenet-r}_report.json），
  GPU 冒烟贯通（--limit-tasks 1 --epochs 1 --local-epochs 1，exit 0，
  task-0 acc 10.47%，metrics 树与 RunManager 格式一致，
  smoke 输出 output/cifar100/fedprotip/smoke_l1/seed_42/）
- 实际设计（与下方旧计划不同，以代码 docstring 为准）：
  1. _map_new_class_index → 到达序重映射（train+test targets）
  2. DataManager.__init__ wrapper：_increments=[到达计数] + test 数组
     过滤成 test_pool（保持官方顺序的有序子集）
  3. partition_data → manifest 版（server 调用点外科手术传入
     cur_task/_dm）；user_groups = inner.train(c,t) 中"t 时刻新类"样本在
     任务新类 DummyDataset 里的位置；已到达类样本 drop 并计数
  4. IncrementalNet.generate_fc 源码 patch：fc 块宽 args['increment']
     → _FPT_STATE['increments'][_cur_task]（方法源内锚点 **3 处**，
     DR 变体行带 bias=False 不含完整子串）
  5. Server_FedProTIP.train 源码 patch：partition 调用点 + 边界 eval
     插入锚点 `self._known_classes = self._total_classes`（dedent 后
     8 空格缩进）→ _fpt_boundary_eval（RNG-neutral，per-(c,k≤t) 在
     test_pool 上按 mask(c,k) 评估，logits 截断到已到达类）
  6. Imagenet_R.download_data 整替（官方 isfile 检查目录恒 False 会
     每次重下载 + 外部 txt 切分 → 改读本地 local_datasets/imagenet-r
     排序遍历）；CIFAR 用 data_dir=local_datasets
  7. stub：wandb/progress（env 未装，代码仅 import）；timm.layers
     （env 是 timm 0.6.7 无 timm.layers，models/vit.py import
     resample_abs_pos_embed —— shim 带真实现，resnet18 路径不会调用）
- 协议映射：epochs=5 轮/任务、local_epochs=30、frac=1.0、classIL=True、
  class_shuffle=0、seed=manifest seed；method-native：resnet18 pretrained、
  SGD lr=0.01 wd=5e-4、batch=64、threshold=0.775
- 冒烟：--limit-tasks 1 --epochs 1 --local-epochs 1（注意：CUDA 初始化
  会被 Trae 沙箱挡住 Error 304，须在沙箱外跑 GPU 训练）

### hepco-fed（external_baselines/hepco-fed，FedCIL，ViT-B/16+L2P+生成器KD）
- ✅ 已完成（2026-11-11）：run_hepco.py + gate_hepco.py，双 manifest
  gate 全 PASS（gate_hepco_{cifar100,imagenet-r}_report.json，114/117 项），
  CIFAR GPU 冒烟贯通（--limit-tasks 2 --rounds 1 --local-epochs 2，exit 0，
  0.17h，task-2 acc 34.7 / plastic 58.2 / lastacc 15.7，metrics 树与
  RunManager 格式一致，client_0_accuracy.csv: (0,0,0)=70.16 (1,0)=22.03
  (1,1)=72.34；drops.json 538；**INR GPU 冒烟也贯通**：0.07h，
  task-2 acc 34.38 / plastic 53.24 / lastacc 13.40，drops=0（INR 全私有
  符合设计），metrics 树正确）
- ⚠️ 运维教训：INR 冒烟首次记录出现 (c,0,0) 双份 identical 行——
  排查确认是被打断的命令留下孤儿进程先跑完了 task 0（18:45-18:49，
  训练+5 条记录+存模型），本轮 run 走 hepco 官方 resume 语义 load 模型
  重评。不是 adapter bug；records.jsonl append-only + latest-wins 是
  预期行为。**重跑前先 `pgrep -f run_hepco` 查孤儿进程**。另外
  `run_in_background=true` 的命令会落回沙箱（CUDA 304），GPU 训练
  只能用前台 + dangerouslyDisableSandbox（超时自动转后台仍是沙箱外）
- 实际设计（以代码 docstring 为准）：
  1. 调和方案 = FedProTIP 首次到达方案的移植：tasks/tasks_real 直接用
     到达位置区间（不是 raw 类！），class_mapping = raw→到达位置，
     add_valid_output_dim=inc[t]；head 本来就是全宽 nn.Linear(768,
     nb_classes)，列选择/CE 目标/prompt 任务推进全部在到达位置空间一致
  2. iIMAGENET_R.load 整替（官方 glob 平铺布局不存在 + random.seed(0)
     shuffle 80/20 → 读 local_datasets/imagenet-r 排序域；兼保
     原 Trainer.__init__ 在 INR 上能活着跑完）
  3. Trainer.__init__ wrapper：原 init 原样跑完（其产物被丢弃）→
     tasks/server_config/num_tasks/task_names 重写 → 11 个数据集重建为
     ManifestDataset（子类 IMBALANCECIFAR/IMBALANCEINR，继承
     __getitem__/__len__/append_coreset）→ server learner 用调和后
     config 重建（prompt_param=[nt,[100,20,1,-1]]）
  4. ManifestDataset：train archive[t]=inner.train(c,t)∩新类（轮转
     公共类首次到达 drop，drops.json）；server test archive[t]=test_pool
     ∩t 新类（pool 序）；client test archive[t]=test_pool∩mask(c,t)；
     load_dataset 覆写（train 分支填 label_counts，官方 raw-id 域，
     train_generator/knowledge_distillation 原生消费）
  5. Trainer.train 源码 patch：lastacc anchor 后插
     `if r == self.n_rounds - 1: _HEPCO_BOUNDARY_EVAL(self, i)`
     → per-(c,k≤t) test_pool∩mask(c,k)，logits 截断到达位置，
     metrics_recorder 落盘（RNG-neutral）
  6. 封章 bugfix ①（官方仓库自带矛盾）：kd.py L93 传 prev_server=，
     prompt.py L30 DualPrompt.update_model 签名却是 prev_task_model=
     （仅 loss_type=='triplet' 分支引用）→ 运行时加 prev_server 别名
     （恢复官方语义意图，冻结跑 loss_type=None 零影响）；gate C1 锚点
     + C2 liveness 双断言
  7. 封章 patch ②：NormalNN.cuda/DualPrompt.cuda `> 1`→`>= 1`——官方
     2 卡脚本恒走 DataParallel 且 train_generator/knowledge_distillation
     硬编码 self.server.model.module.prompt；单卡 DataParallel 纯包装
     数值等价，保 .module 可用；CPU（gate，gpuid=-1）不调 cuda 无影响
  8. stub：wandb（未装，init/log noop）+ fairscale（未装，
     checkpoint_wrapper 直通，ckpt_layer=0 永不真用）；timm augreg
     ViT-B/16 npz 已在 ~/.cache/torch/hub/checkpoints（离线可载）
- 协议映射：n_clients=5、n_rounds=5（冻结；官方脚本 10）、schedule=[30]
  （冻结；官方 10）、seed=manifest seed、max_task=5；method-native 其余
  （learner=prompt/L2P weighted_l2p、prompt_param [100,20,1,-1]、--kl
  --hepco、λ_KL=1、λ_mse=0.1、noise_dim=64、replay_ratio=.5、memory=0
  **无 coreset 回放**、Adam lr=0.005、batch 64、top_k=1、kd_epochs=200、
  generator_epochs=100）
- 冒烟注意：--local-epochs 必须 ≥2（K-1 做余弦分母，=1 除零）；
  limit-tasks=1 时 hepco 的"最后任务 schedule 减半"逻辑会把训练砍没，
  冒烟用 limit-tasks 2
- 输出：output/{cifar100|ImageNet-R}/hepco/manifest_adapter/seed_42/

### GLFC（external_baselines/GLFC，CVPR 2022 Dong et al.，rehearsal）
- run_glfc.py + gate_glfc.py 已完成（2026-11-11）：双 manifest gate 全 PASS
  （gate_glfc_{cifar100,imagenet-r}_report.json，CIFAR 92/92、INR 91/91）
- 实际设计（以代码 docstring 为准）：
  1. fl_main.py 是平铺脚本（无 main()）→ 读源码做锚定替换后 exec（hepco
     Trainer.train 源码 patch 的同款模式）；GLFC 模块类级 patch 在 exec 前
     完成，`from GLFC import GLFC_model` 拿到的即 patched 类
  2. 调和方案 = FedProTIP 首次到达方案移植：arrival position 空间，
     CIFAR increments [32,26,14,16,12]、INR [40]*5；beforeTrain 整替为
     manifest 版（numclass=cumulative K_t、current_class=mask(c,t) 到达位
     ——官方 random.sample 6 类分区机制被 manifest 替换；task_size=inc[0]
     使 train() 的首任务检查 numclass==task_size 保持原语义）
  3. 数据集整替为 duck-type（不继承 iCIFAR100）：ManifestClientDataset×5
     （官方 125 个客户端共享一个全局 trainset —— GLFC 的"客户端"是模拟
     概念；manifest 是 per-client 的，所以每个 GLFC_model 拿自己的
     client-scoped 数据集对象）+ ManifestTestDataset（test_pool 评估池，
     getTestData([0,K_t)) 到达位区间语义）；getTrainData classes=到达位、
     exemplar 拼接保官方顺序；get_image_class 返回该客户端 inner.train
     自有样本（官方语义 = 任意客户端可读全类样本，manifest 限定所有权）
  4. 客户端流转整块中立化（frozen：固定 5 客户端全员参与）+ 参与
     random.sample→全选 + 头部增长 classes_learned+=increments[task_id]
     + 初始宽度 network/proxyServer/classes_learned=K_0 + eval 调用换
     _GLFC_GLOBAL_EVAL（官方 shuffle=True 全局评估语义 + 任务末轮
     RNG-neutral per-(c,k≤t) 边界评估，logits 全宽=K_t 无需截断）
  5. **轮转公共类零 drop**：GLFC 是 rehearsal 方法（全宽 BCE+蒸馏+exemplar
     回放），已到达类的样本合法进入训练（列已存在、可继续学习），
     drops.json total=0；与 FedProTIP/hepco 的 first-arrival drop 不同，
     论文行加 †uses replay（exemplar memory + 梯度反演回放）
  6. **新发现的 manifest 边界：饥饿类**——mask 类在该客户端 inner.train
     可能为 0 样本（CIFAR 恰好 2 例：(c0,t4,raw10) 唯一样本落 inner.test、
     (c2,t3,raw56) outer 0 样本）。任何方法都无法训练 0 样本类，且 GLFC
     的 exemplar 构造会崩 → beforeTrain 过滤之并计入 drops.json 的
     zero_train_sample_classes_omitted（gate B3c 断言恰为此 2 例）；其列
     由到达任务的持有客户端正常学习，边界评估不受影响（test_pool 有样本）
  7. 封章 bugfix ①：GLFC_model.train LR 调度硬编码 `ep_g * 20`×3
     （epochs 是构造参数却被无视）→ 源码 patch `ep_g * self.epochs`
     （官方 epochs=20 时数值等价；frozen 协议 local epochs=30 需要）；
     %200 衰减周期硬编码保留（官方算术，文档记录）
  8. 封章 patch ②（仅 INR）：LeNet 编码器与 ProxyServer.reconstruction
     硬编码 100 类（fl_main LeNet(num_classes=100)、reconstruction
     np.zeros((1,100)) + range(100)）→ manifest nb_classes（CIFAR=100
     数值不变）
  9. INR 桥接：无官方 INR 支持 → 预解码 local_datasets/imagenet-r
     （walk_imagenet_r 域）并统一 resize 到 84×84（GLFC 自家的
     Mini_Imagenet 惯例：84×84 数组 + 官方 RandomCrop(32,pad4)/
     Resize(32) transforms 原样保留）→ 全管线留在 GLFC 32×32 设计空间
     （LeNet hideen=768、(1,3,32,32) 反演 dummy、CIFAR 归一化常数全不动）
  10. cv2 未装（fl_main 无条件 import mini_imagenet→cv2）→ noop stub
      fail-loud；matplotlib→Agg；无 wandb
- 协议映射：5 客户端 × 5 任务、5 轮/任务（官方 10）、30 local epochs
  （官方 20，见封章①）、batch 128、memory_size 2000、seed=manifest seed；
  method-native 其余（resnet18_cbam from scratch、SGD lr=2.0 wd=1e-5、
  200 步 LR 衰减、entropy-signal、prototype 梯度共享、DLG 反演
  250 LBFGS×20 图/类）
- 冒烟注意：GPU 训练必须沙箱外 + 绝对路径 python；代理反演按类数×客户端
  数线性开销（40 grad×250 LBFGS iter），冒烟单轮也可能数分钟
- **【2026-11-12 冒烟实况补充（run_glfc.py 三处追加）】**：
  1. L471 `if self.TrainData != []`（numpy 数组 elementwise 比较）→
     `if len(self.TrainData)`（消 DeprecationWarning 刷屏，数值等价）
  2. **num_workers=0 封章（环境级，非官方 bug）**：官方
     `_get_train_and_test_dataloader` 的 `num_workers=8` 在父进程已初始化
     CUDA 后 fork 8 个 loader worker，与父进程 CUDA/pin-memory 线程
     概率性死锁（r2 实测 4/5 客户端过、第 5 个永卡 loader 迭代器，8 条
     worker warning 一条不出现；CUDA 非 fork-safe）。方法级 getsource
     封章替换为 num_workers=0（数据加载回主进程，数值完全一致），r4
     验证 5 客户端训练+exemplar+FedAvg 全通过
  3. **proxy 反演开销实测与修正（官方设计，非 adapter 引入）**：
     - 反演**不是每轮跑**——`proto_grad_sharing` 仅 entropy signal=True
       时返回梯度，否则 pool_grad 空 → reconstruction 整段跳过
       （Fed_utils L49 的 print(signal) 即日志里的 True）。signal 触发
       ≈ 首轮（last_entropy=0 恒触发）+ 每任务边界首轮（新类熵跳变
       >1.2）→ 冻结跑（5 任务 25 轮）仅 ~6 个反演事件，非 25 轮全跑
     - 每事件 pool_grad = 5 客户端 × mask 类 ≈ 32 条（官方 10 客户端
       ×6 类=60 条）；每条 250 LBFGS 外层 × ~21 closures = 5250 次
       LeNet 双反传 closure（tmp/bench_glfc_recon.py 微基准，逐行复刻
       官方 closure）：
       **GPU（本机，5 任务共卡）282ms/closure → 24.7 分/条；
         CPU 同一 closure 13.2ms → 1.2 分/条（21×，launch-bound 反例）**
       r4 实测 ≥32 分/条与之吻合
     - 修正后正式跑估算：GPU ≈ 6 事件 × 32 条 × 25 分 ≈ 3-4 天/数据集
       （此前"2.5 周"高估 5×）；若 proxy 反演封章重路由到 CPU
       （数学恒等、零截断、零语义变化）≈ 3-4 小时/数据集
     - **封章策略已裁决并实施（C'，用户批准 2026-11-12）**：proxy
       反演重路由到 CPU。patch_proxy_server 扩展为双封章（nb_classes
       参数化 + C' 设备重路由 5 锚点：randn/label_pred/criterion/
       model_to_device→recon_model.cpu()/grad_truth_temp 逐条 .cpu()），
       gate 加 B12b（补丁侧：无 .to(self.device)、含 .cpu() 标记）与
       C3c（官方侧：3 个设备锚点 verbatim 计数），双数据集 gate
       94/94、93/93 PASS；meta.json 记 proxy_recon_device 字段
     - **顺带修复 gate 抓出的回归**：上会话追加 num_workers 封章时
       `glfc_mod.GLFC_model.train = ns["train"]` 回绑行意外丢失
       （train patch exec 后未绑定回类，B10 FAIL）——已补回；r3/r4
       冒烟当时跑的是官方原版 train（冒烟轮数少未触 LR 衰减边界，
       无数值影响，但正式跑前必须修复，现已闭环）
     - **C' 冒烟验证（smoke_l2r1e1_cpu，limit 2 任务×1 轮×1 epoch）**：
       0.73h 全程贯通（GPU 原样同负载 ~17h，~23×）；CPU 反演吃满
       ~11 核（55s/条），收敛质量正常（current_loss ~1e-6）；R 矩阵
       R[0][0]=6.09 / R[1][0]=0.00 R[1][1]=7.56，records 15 条，
       drops=0（rehearsal 符合设计），metrics 树完整
     - 正式跑估算更新：训练段（30 epochs×25 轮）为主 ~16h + proxy
       ~4h（6 事件×35-40 分）≈ **~1 天/数据集**


### Powder/FedPCR（✅ 完成：run_powder.py + gate_powder.py 双数据集 gate
### 79/79 PASS + CIFAR & INR GPU 冒烟双贯通）
- 仓库就位：external_baselines/Powder/（codeload tarball + wget 断点续传
  方案成功，clone 7+ 次失败后改用 tarball）。ICML 2024 "Federated
  Continual Learning via Prompt-based Dual Knowledge Transfer"
- 官方命令（powder_job.slurm，ImageNet-R）：
  `python fl_main_hard.py --device 0 --global_update_lr 100 --img_size 224
  --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0
  --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5
  --learning_rate 0.005 --prompt_flag codap_2d_v2
  --method cprompt_powder_v2 --batch_size 64
  --prompt_param 25 10 10 8 0 0 6 10 8 --epochs_local 10 --seed 2021`
  → method=cprompt_powder_v2（ViT-B/16 prompt 方法，无 proxy server、
  无 exemplar 训练）。**CIFAR 无官方命令**（论文只跑 INR/DomainNet）
- 代码结构（src/ 887 行 fl_main_hard.py 平铺脚本，同 GLFC 用 exec 模式）：
  - class_distribution_client[c][t]（position 空间）/_real（raw 类）/
    _proportion（[start,start+0.2] 切片）三结构 = manifest 注入点；
    官方生成块 L66-232 **没有 CIFAR 分支**（CIFAR 原生 NameError，
    全块替换为 manifest 派生三结构，fail-loud 锚点封章）
  - prompt_hard.py DualPrompt_hard.beforeTrain(task_id, group, classes,
    client_index, global_task_id_real, class_real)：从 model.class_
    distribution 取 mask；group=0 停留旧任务/churn 语义、group=1 前进新
    任务；getTrainData hard 签名 (classes, [], [], client_index,
    classes_real=…, classes_proportion=…, class_distribution_client_di=…)
    —— 官方 iCIFAR100.getTrainData 不收这些 kwargs（CIFAR+cprompt 原生
    TypeError）；manifest duck-type 实现 hard 全签名即可承接，官方文件
    不动
  - iDOMAINNET.getTrainData（INR 语义）：按 raw 类选数据、proportion 切
    片、标签=position（classes[classes_real.index(label)]）；
    getTestData_hard(classes, classes_real) 同样 raw 选数据、position
    打标签——与首次到达 position 调和天然兼容
  - **非均匀 increments 天然支持**：Incremental_learning(task_id) 只设
    task_id+process_frequency，head 一次建满 numclass；输出屏蔽靠
    set_global/client_class_min_output 类列表（非 task_size 乘法）——
    比 GLFC 干净，无 head 封章；CodaPrompt_2d_v2 prompt 池只由
    prompt_param 决定，n_tasks 不影响池大小
  - churn：cprompt_powder_v2 非 extension → 每任务边界 50% 客户端
    random.sample 成 old_client_0（beforeTrain group=0 停旧任务）；
    统一协议封章：`old_client_1 = random.sample([i for i in
    range(overall_client)], int(overall_client * 0.5))`（L521 唯一锚）
    → 全量列表，全客户端同步前进，PartA 记录官方差异
  - 官方 eval（model_global_eval_hard，Fed_utils.py L1139）：
    test_dataset.getTestData_hard(global_class_output, …real) 在
    已到达类池上按 (client,task) 模型评估；fl_main L865 else 分支为
    注入点——包装 _POWDER_GLOBAL_EVAL：官方语义原跑 + 任务末轮
    (ep_g+1)%tasks_global==0 时做 manifest boundary eval（model_g 对
    每 (c,k≤t) test_pool∩mask(c,k)，RNG-neutral shuffle=False，同
    GLFC _glfc_boundary_eval；一次/边界去重）
  - nni.get_next_parameter()/report_*：fl_main 顶部调用（slurm 裸跑
    也是 nni 实验）；sys.modules 注入 stub（get_next_parameter→{} 即
    默认 params：topk_for_task_selection=3 等，喂 prompt topk），
    无需源码 patch
  - cv2 缺失（proxy_data.py import cv2 → ProxyServer → fl_main）：
    sys.modules stub（同 GLFC）；timm 0.6.7 vit_base_patch16_224
    实际引用 **augreg npz**
    （B_16-i21k-300ep-...res_224.npz，~346MB，缓存已就位并验证可
    加载）——jx_vit_base_p16_224-80ecf9dd.pth 不在此 timm 的加载
    路径上，无需下载
  - transform（dataloaders/utils.get_transform，官方不动）：train=
    RandomResizedCrop(224)+Flip、CIFAR test=Resize(224)、INR test=
    Resize(256)+CenterCrop(224)；mean=(0,0,0) std=(1,1,1)；
    dataset_stats 有 CIFAR100 键（size=32，无 KeyError）；CIFAR 32×32
    原生数组直接喂（RandomResizedCrop 上采样），无需 GLFC 式 84×84
    桥接；INR 沿用 GLFC 84×84 预解码惯例（transform 自行 resize 224）
- 配置决策（CIFAR 无官方先例 → 沿用 INR 官方超参，仅协议字段适配）：
  numclass=K_final(100/200)、class_per_task/task_size=inc[0](32/40)、
  tasks_global=R（每任务轮数，同 GLFC 映射）、epochs_global=R×T、
  num_clients=5=local_clients、epochs_local=10、batch_size=64、
  learning_rate=0.005、global_update_lr=100、prompt_param=25 10 10 8
  0 0 6 10 8、prompt_flag=codap_2d_v2、method=cprompt_powder_v2、
  img_size=224、seed=manifest seed、device=0
- manifest 注入结构（✅ run_powder.py 已实现，镜像 run_glfc.py 骨架）：
  class_distribution_client[c][t]=arrival positions(mask(c,t))、
  _real[c][t]=raw 类、_proportion=[0,1]（duck-type 按自有样本精确
  所有权，忽略 proportion）；class_real/global_task_id_real=identity
  （cprompt_powder_v2 不含 classincremental/extension/extencl）；
  ManifestClientDataset.getTrainData(classes, exemplar_set,
  exemplar_label_set, client_index, classes_real=None,
  classes_proportion=None, class_distribution_client_di=None,
  exe_class=None) 按 raw 类取本客户端 inner.train 自有样本、标签打
  position；ManifestTestDataset.getTestData_hard(positions, raw) 按
  test_pool；训练日志落 output 树 training_log/（P5 封章重定向）+
  metrics 走 recorder
- **实现纪要（run_powder.py 完成后新增的发现与封章）**：
  - **S1 封章**（设计时未发现的阻断 bug）：CodaPrompt_2d_v2.__init__
    中 weight/weight_c/task_embedding 尺寸硬编码
    `int(numclass/class_per_task)`（均匀任务假设：官方 500/20=25 行
    恰好=e_task_number；CIFAR 100/32=3 行 → IndexError）。运行时方法
    级 rebind 改为 `e_task_number`（官方配置数值等价）
  - **S2 封章**：DualPrompt_hard._ntd_loss else 分支（cprompt 路径）
    用 `refine_as_not_true(numclass)` + `[first:last]` 切片限制蒸馏列
    ——仅当 current_class 连续升序时等价；manifest 散列 positions
    会静默蒸馏错误列。改用官方作者自己写好的 refine_as_not_true2
    （直接索引，连续场景数值等价已验证）
  - 方法级 exec rebind 的 `super()` 零参形式缺 `__class__` cell →
    `_patch_method` 统一重写为两参形式（数值等价）
  - 饥饿类（0 owned inner.train 样本 → 官方 per-class getTrainData
    torch.mean(None) 崩溃）：CIFAR 恰 2 例 (c0,t4,raw10)/(c2,t3,
    raw56)——其列均由 c1 在到达任务正常学习（218/16 样本），结构级
    过滤 + drops.json 记账；INR 0 例
  - 轮转公共类合法再训练（全宽 fc + 当前类 CE + 每轮 ntd 蒸馏）→
    零 drop
  - stubs：nni(→{})、cv2(fail-loud)、fgvcdata(CUB/StanfordDogs/
    StanfordCars/Aircraft 类体 import 期求值哨兵)、pytorch_lightning
    (LightningDataModule+动态兜底)、numba(njit 恒等装饰器)
  - P1-P6 fl_main 锚点封章（partition 三结构注入、数据集分支、churn
    全量、参与全量、training_log 重定向、global eval 包装×2）；
    L700 per-client 调用属死分支保留原样
  - GPU 冒烟（CIFAR，limit 2 任务×1 轮×1 epoch×5 客户端，batch 16）：
    S2 蒸馏路径实测运行，final global acc 82.86%，R 矩阵
    R[0][0]=0.78 / R[1][0]=0.61 R[1][1]=0.58（5 客户端均值），
    metrics 树完整；after_task 0-based 与其他 adapter 一致
  - GPU 冒烟（INR，2026-11-12，同参）：贯通，final 70.96%，
    R[0][0]=0.65 / R[1][0]=0.50 R[1][1]=0.41（avg 0.4577，bwt -0.1494），
    records 15 条（2 边界×(5+10) 组合），drops=0（INR 全私有符合设计），
    显存 ~3.3GB，输出 output/ImageNet-R/powder/smoke_l2r1e1/seed_42/

## 实现顺序与产出
1. ✅ run_fedprotip.py + gate_fedprotip.py（CIFAR+INR gate 双 PASS）
2. run_hepco.py + gate（INR 有官方脚本，可能两数据集都能做）
3. run_glfc.py + gate（CIFAR 先；INR 走主仓桥接）
4. ✅ run_powder.py + gate_powder.py（CIFAR+INR gate 双 79/79 PASS；
   CIFAR + INR GPU 冒烟双贯通）
5. ✅ 命令文档：RUN_BASELINES.md（repo 根；izfree 机器口径：训练命令
   直出+自检；含 GLFC proxy 开销警告与封章策略待定项）
   输出树 output/{dataset}/{method}/manifest_adapter/seed_{seed}/

## 关键先例文件（写 adapter 前必读）
- protocol_adapters/run_c2p.py（class-IL 全局模型调和 + OBSERVE 评估 +
  任务推进规则替换 + nni stub + 每 (client,after,test) 记录怎么来）
- protocol_adapters/manifest_splitter.py（FedMGP：spliter/get_data/inner
  三注入点 + INR 主仓桥接 + 封章 bugfix 模板）
- protocol_adapters/metrics_recorder.py（resume-safe 记录器）
- protocol_adapters/gate_fedmgp.py（gate 三段式）
