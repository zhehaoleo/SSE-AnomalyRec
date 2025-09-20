# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def _pair_sqdist(a, b):
    # 欧式平方距离，逐样本
    return torch.sum((a - b) ** 2, dim=1)

def _pair_dist(a, b, eps=1e-8):
    # 欧式距离，逐样本
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1) + eps)

class ActionAwareLoss(nn.Module):
    """
    动作邻域加权的插值 + Triplet 联合损失。

    - x1,x2,x3,y1: (B,D) 分别对应 Q_{i-1}, Q_i, Q_{i+1}, 负样本 的向量表示
    - s: (B,) 三连问的一致性权重 ∈ [0,1]，表示 (i-1,i,i+1) 是否“同一动作内”
         s 越大 -> 同一动作可能性越高 -> 插值/对比越强
    - alpha: 插值损失中 "三者两两距离" 的系数（对应论文 Eq.1 的第二部分）
    - margin: triplet 的间隔
    - beta: 联合损失里 triplet 的权重（对应论文 L = l1 + beta * l2）
    - triplet_mode: "same_action_stronger"（同动作更强）或 "cross_action_stronger"（跨动作更强）
    """

    def __init__(self, alpha: float = 0.2, margin: float = 0.2, beta: float = 1.0,
                 triplet_mode: str = "same_action_stronger"):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.beta = beta
        assert triplet_mode in ("same_action_stronger", "cross_action_stronger")
        self.triplet_mode = triplet_mode

    def forward(self,
                x1: torch.Tensor,  # (B,D)
                x2: torch.Tensor,  # (B,D)
                x3: torch.Tensor,  # (B,D)
                y1: torch.Tensor,  # (B,D)
                s: torch.Tensor    # (B,)
                ) -> torch.Tensor:
        # —— 加权插值（论文 Eq.1 的思想，乘以 s —— 同一动作更强） ——
        mid = 0.5 * (x1 + x3)                      # (B,D)
        loss_mid = _pair_sqdist(x2, mid)           # (B,)

        pair12 = _pair_sqdist(x1, x2)
        pair23 = _pair_sqdist(x2, x3)
        pair13 = _pair_sqdist(x1, x3)
        pair_sum = pair12 + pair23 + pair13        # (B,)

        interp_each = loss_mid + self.alpha * pair_sum
        # s=0 -> 不插值；s=1 -> 全量插值
        interp_weighted = s * interp_each
        l1 = torch.mean(interp_weighted)

        # —— Triplet（anchor=x1, positive=x3, negative=y1），按 s 可选加权 ——
        d_ap = _pair_dist(x1, x3)                  # (B,)
        d_an = _pair_dist(x1, y1)                  # (B,)
        trip_raw = torch.clamp(d_ap - d_an + self.margin, min=0.0)  # (B,)

        if self.triplet_mode == "same_action_stronger":
            w_trip = s
        else:  # "cross_action_stronger"
            w_trip = 1.0 - s

        l2 = torch.mean(w_trip * trip_raw)

        # —— 总损失 —— 对齐论文 L = l1 + beta * l2（这里只是 l1 换成了加权版）
        total = l1 + self.beta * l2
        return total
