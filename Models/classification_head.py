from copy import deepcopy

import torch
from torch import nn


class Chead(torch.nn.Module):
    def __init__(self,label_num,dropout=0.5):
        """A5-v2 F1（--head_dropout，2026-10-28）：train 端 Dropout p 参数化。
        默认 0.5 = 官方行为逐位一致（构造不消耗 RNG，前向路径不变）；
        p=0.0 为归因消融臂（phase-2 前向不再消耗 dropout RNG，臂内自洽）。"""
        super(Chead,self).__init__()

        # self.head = nn.Sequential(nn.Linear(768*2, label_num))
        self.head = nn.Sequential(nn.Linear(768*2, 512),
        # 激活函数
        nn.ReLU(),
        # Dropout 层，防止过拟合
        nn.Dropout(p=dropout),
        # 第二层全连接层
        nn.Linear(512, label_num))
    def forward(self,x):
        x = self.head(x)
        return x

    def load_head(self,head):
        self.head = deepcopy(head)

    def get_head(self):
        return self.head


class StandardizedChead(torch.nn.Module):
    """A5-v2 F3（--head_standardize）refit 头：train-split 标准化输入 +
    Chead 形状 MLP。mu/sd 作为 persistent buffer 随 state_dict 存取，
    resume 时按 keys（mean/std + head.*）与 Chead 区分（rebuild_client_head）。
    forward 语义与 Chead 完全同构（输入 1536 维 x1，输出 label_num logits），
    因此可无缝进入 Tail_Anchor.load_head / evaluate 路径。"""

    def __init__(self, label_num, mean, std, hidden=512, dropout=0.1):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone().float())
        self.register_buffer("std", std.detach().clone().float())
        self.head = nn.Sequential(nn.Linear(mean.shape[0], hidden),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(hidden, label_num))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.head(x)

    def load_head(self, head):
        self.head = deepcopy(head.head)
        self.mean = head.mean.clone()
        self.std = head.std.clone()

    def get_head(self):
        return self.head
