from copy import deepcopy

import torch
from torch import nn


class Chead(torch.nn.Module):
    def __init__(self,label_num,dropout=0.5):
        """Dropout p 参数化：默认 0.5 = 官方行为逐位一致（构造不消耗
        RNG，前向路径不变）；A5-v2 M2' refit 头配方使用 p=0.1。
        （F1 --head_dropout 消融已随判死方向清理删除，存档见 git 历史。）"""
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
