import time

import torch
import torch.nn as nn

class Global_Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 route_fix=False, route_tau=0.1, route_seed=0):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        # A2-R 路由修复开关（CLI --route_fix，默认 False = 官方行为逐位一致）：
        #   官方 FedTA 的 Global_Prompt 路由存在结构性死锁——top-k/argmax 硬选择
        #   不可导（路由决策无梯度）+ uniform(-1,1) 随机初始化近似平行 + pull 只拉
        #   选中槽 → 初始垄断正反馈固化，所有样本 100% 命中单槽（诊断证据：
        #   analyze_prompt_signal.py 实测 keyH≈0.004 bits），prompt 池退化为固定
        #   bias。微训练探针 diagnostics/probe_route_fix.py（已验证）判决 R1+R2
        #   使路由 5/5 客户端复活（keyH +3~4 bits）且准确率守住。
        #   route_fix 开启 = R1（构造处正交球面 key 初始化）+ R2（forward 软路由）。
        #   关闭时两个分支均不执行、不消耗任何 RNG（全局 torch 流零扰动）。
        self.route_fix = route_fix
        # R2 软路由温度 τ：w = softmax(similarity / route_tau)，CLI --route_tau
        self.route_tau = route_tau

        # 初始化提示池  提示数量x长度x嵌入维度
        # 提示长度为5？
        # 这里直接使用prompt池
        prompt_pool_shape = (pool_size, length, embed_dim)
        if prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)

        # 提示健初始化
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                if route_fix:
                    # R1：正交球面初始化（--route_fix 开启分支；探针
                    # diagnostics/probe_route_fix.py 的 reset 臂
                    # （reset_prompt_key_orthonormal）同款语义）。
                    # 官方 uniform(-1,1) 随机 key 近似平行且选择决策不可导，
                    # 初始 monopoly 正反馈固化为单槽命中（死锁第 2 环节）；
                    # 正交化后初始方向两两垂直（球面 Haar 均匀），从源头打破。
                    # 数学上与 nn.init.orthogonal_ 等价（高斯矩阵 QR 分解），
                    # 但改走独立 torch.Generator：既不消耗全局 RNG 流
                    # （数据划分 / 训练流与关闭时不发生相对移位），又可复现
                    # （route_seed = run 的 --seed）。
                    if pool_size > embed_dim:
                        raise ValueError(
                            f"route_fix R1 orthogonal init needs "
                            f"pool_size<=embed_dim, got {pool_size}>{embed_dim}")
                    gen = torch.Generator()  # 独立随机流，不碰全局 torch RNG
                    gen.manual_seed(int(route_seed))
                    raw = torch.randn(embed_dim, pool_size, generator=gen)
                    q, _ = torch.linalg.qr(raw)   # (dim, pool) 列正交
                    rows = q.t().contiguous()     # (pool, dim) 行正交
                    # forward 中 key 与 cls 特征都过 l2_normalize，
                    # 绝对尺度无信息、方向是全部信息 → 按行 L2 归一化落位
                    self.prompt_key = nn.Parameter(self.l2_normalize(rows, 1))
                else:
                    # 官方路径（route_fix 关闭）：逐字保留，全局 RNG 消耗
                    # 次数与原实现一致 → 默认 run 逐位可复现
                    self.prompt_key = nn.Parameter(torch.randn(key_shape))
                    nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        # 可以用class-specific prompt吗？
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # x_embed_norm 就是key
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C    先是随机的
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
            # print(f"X_embed_norm: {x_embed_norm.shape}")

            # 计算相似度 查询prompt
            # 直接embedding与prompt_key相乘,得到的是相似度矩阵
            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size

            # prompt_mask其实就是判断哪些prompt要选择
            # 所以没有prompt_mask就不用余弦相似度了来进行选择了
            if self.route_fix:
                # -----------------------------------------------------------
                # A2-R R2：可导软路由（--route_fix 开启分支）
                # 探针（diagnostics/probe_route_fix.py）判决：R1+R2 使路由
                # 5/5 客户端复活（keyH +3~4 bits）且准确率守住；训练与推理
                # 同款软加权（不做 train-soft/infer-hard 分裂，探针验证的
                # 就是全软形态）。官方 top-k/argmax 硬选择不可导是死锁第 1
                # 环节，这里替换为全池可导 softmax：w = softmax(sim/τ)。
                # τ=route_tau=0.1 时 softmax 的有效支撑集中在相似度最高的
                # 少数槽上，与 top-k 选择语义兼容且对 prompt_key 处处可导。
                # -----------------------------------------------------------
                if prompt_mask is not None:
                    # 探针同款 fail-closed：软路由与 use_prompt_mask 组合
                    # 未经验证（本协议 use_prompt_mask=False），拒绝静默不一致
                    raise RuntimeError(
                        "--route_fix soft routing is incompatible with "
                        "use_prompt_mask=True")
                # 按样本软加权，天然取代官方 batchwise 批内多数投票分支
                w = torch.softmax(similarity / self.route_tau, dim=-1)  # B, Pool
                batched_prompt = torch.einsum('bp,plc->blc', w, self.prompt)  # B, length, C
                if self.top_k > 1:
                    # token 数保持 length×top_k 不变（结构兼容：官方聚合 /
                    # pos_embed / head_type='prompt' 消费者不受影响）
                    batched_prompt = batched_prompt.repeat(1, self.top_k, 1)
                out['prompt_idx'] = torch.argmax(similarity, dim=1)  # 仅诊断观察用（不可导）
                out['prompt_norm'] = prompt_norm
                out['x_embed_norm'] = x_embed_norm
                out['similarity'] = similarity
                out['soft_weights'] = w
                # 软路由版 pull 项（探针同款公式）：
                #   官方 reduce_sim = mean_b(Σ_{k∈top-k} cos(x_b, key_sel))
                #   软版本        = top_k · mean_b(Σ_s w[b,s]·cos(x_b, key_s))
                # 软权重和为 1，乘 top_k 对齐官方 top-k 求和的量级；对
                # prompt_key 经 w 可导 → CE − 0.1×pull 官方语义保持成立，
                # 无需退化为纯 CE。
                reduce_sim = (self.top_k * torch.sum(w * similarity)
                              / x_embed.shape[0])
                out['reduce_sim'] = reduce_sim
            else:
                # 官方路径（route_fix 关闭）：以下代码逐字保留，保证默认
                # run 的前向与原实现逐位一致
                if prompt_mask is None:
                    # 选择topk个
                    _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                    if self.batchwise_prompt:
                        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                        # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                        # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                        # Unless dimension is specified, this will be flattend if it is not already 1D.
                        if prompt_id.shape[0] < self.pool_size:
                            prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                            id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                        _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                        major_prompt_id = prompt_id[major_idx] # top_k
                        # expand to batch
                        idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
                else:
                    idx = prompt_mask # B, top_k
                    print(idx)

                # batched_prompt_raw 就是topk的prompt
                batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
                batch_size, top_k, length, c = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

                out['prompt_idx'] = idx

                # Debugging, return sim as well
                out['prompt_norm'] = prompt_norm
                out['x_embed_norm'] = x_embed_norm
                out['similarity'] = similarity

                # Put pull_constraint loss calculation inside
                batched_key_norm = prompt_norm[idx] # B, top_k, C
                out['selected_key'] = batched_key_norm
                x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
                sim = batched_key_norm * x_embed_norm # B, top_k, C
                reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

                # print('-------------------')
                # print(x_embed_norm.shape)  #
                # print(batched_key_norm.shape)
                #
                #
                # print(sim.shape)
                # print(sim)
                # print(torch.sum(sim))
                # print(reduce_sim.shape)
                # print(reduce_sim)
                # time.sleep(50)

                out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]

        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
