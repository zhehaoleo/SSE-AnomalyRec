import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnWindowPool(nn.Module):
    """
    对每个时间步 t，在 [t-W_left, t+W_right] 上做多头注意力，得到 ê_t。
    - e_seq: [L, D_in]   （单样本的时间序列，已去掉 pad）
    - ctx_seq: None      （需要条件注意力时可以加 context，这里先留空）
    返回: [L, D_out]
    """
    def __init__(self, D_in, heads=4, W_left=3, W_right=0, out_dim=None, dropout=0.0):
        super().__init__()
        self.W_left  = W_left
        self.W_right = W_right
        self.h = heads
        self.D_in = D_in
        self.D_out = out_dim or D_in
        assert self.D_out % self.h == 0, "out_dim 必须能被 heads 整除"
        d_head = self.D_out // self.h

        # 可学习查询（无上下文版）
        self.q_param = nn.Parameter(torch.randn(self.h, d_head))

        self.Wk = nn.Linear(D_in, self.D_out, bias=False)
        self.Wv = nn.Linear(D_in, self.D_out, bias=False)
        self.out = nn.Linear(self.D_out, self.D_out, bias=True)
        self.dp = nn.Dropout(dropout)

    def forward(self, e_seq: torch.Tensor):
        """
        e_seq: [L, D_in]
        """
        L, _ = e_seq.shape
        H = self.h
        d_head = self.D_out // H

        k_all = self.Wk(e_seq).view(L, H, d_head)  # [L,H,d]
        v_all = self.Wv(e_seq).view(L, H, d_head)  # [L,H,d]
        scale = 1.0 / math.sqrt(d_head)

        outs = []
        for t in range(L):
            s = max(0, t - self.W_left)
            e = min(L, t + self.W_right + 1)
            l = e - s

            # 转成 [H,l,d]
            k = k_all[s:e].transpose(0, 1)  # [H,l,d]
            v = v_all[s:e].transpose(0, 1)  # [H,l,d]

            q = self.q_param  # [H,d]

            # 注意力分数: [H,l]
            att = torch.einsum('hd,hld->hl', q, k) * scale
            w = F.softmax(att, dim=-1)  # [H,l]
            w = self.dp(w)

            # 池化: [H,d]
            pooled = torch.einsum('hl,hld->hd', w, v)
            outs.append(pooled.reshape(-1))  # [D_out]

        outs = torch.stack(outs, dim=0)  # [L, D_out]
        return self.out(outs)  # [L, D_out]
