from copy import deepcopy

import torch
import torch.nn as nn

from Models.classification_head import Chead


# 直接在尾部加参数
class Tail_Anchor(nn.Module):
    def __init__(self,anchor_size,key_size,nb_class):
        super(Tail_Anchor,self).__init__()
        self.size = anchor_size
        self.key_size = key_size
        self.nb_class = nb_class

        key_pool_size = (nb_class,key_size)
        self.key = nn.Parameter(torch.randn(key_pool_size))
        nn.init.uniform_(self.key, -1, 1)

        self.head = Chead(200)
        anchor_pool_size = (nb_class,key_size)
        self.anchor_pool = nn.Parameter(torch.randn(anchor_pool_size))
        # nn.init.uniform_(self.anchor_pool, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    # x是预训练给的768维
    def forward(self,x,class_mask):
        # 先找key
        # x_embed_norm 就是key
        x_embed_norm = self.l2_normalize(x, dim=1)  # B, C

        tem_key = self.key.reshape(-1,self.key_size)

        tem_key = tem_key.squeeze(0)
        # print(x_embed_norm.shape)

        key_norm = self.l2_normalize(tem_key, dim=1)  # Pool_size, C    先是随机的

        similarity = torch.matmul(x_embed_norm, key_norm.t().to('cuda'))  # B, Pool_size

        q,index = torch.topk(similarity,k=1)
        # print(q,index)


        anchor = self.anchor_pool[index]
        # print(anchor.shape)

        anchor = anchor.reshape(-1,768)
        # print(x.shape)
        # print(anchor.shape)
        x1 = torch.stack((x,anchor),dim=1).view(-1,768*2)
        x = self.head(x1)

        # Put pull_constraint loss calculation inside
        batched_key_norm = key_norm[index]
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C
        reduce_sim = torch.sum(sim) / self.key_size  # Scalar

        return x,x1,reduce_sim

    def load_head(self,head):
        self.head = deepcopy(head)

    def get_head(self):
        return self.head







