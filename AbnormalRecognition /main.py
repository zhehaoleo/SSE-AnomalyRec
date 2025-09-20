# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================
# 全局配置
# ==============================
CSV_PATH = "example_live_data.csv"   # 请自行准备与下方列名匹配的 CSV
PAID_COLS = ["全部付费流量", "小店随心推", "品牌广告", "千川品牌", "千川pc版", "其他广告"]
RESPONSE_COLS = [
    "全部自然流量", "成交金额", "实时在线人数", "进入直播间人数",
    "关注", "评论次数", "点赞次数", "新增粉丝数"
]
CONTEXT_COLS = None

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

USE_SINKHORN = True

SAVE_SCALERS = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ================= Sinkhorn（熵正则OT） =================
def sinkhorn_transport(a, b, M, epsilon=0.1, max_iter=50):
    """
    Sinkhorn-Knopp 近似求解熵正则OT:
      min_T <T, M> + ε H(T) s.t. T 1 = a, T^T 1 = b

    a: (batch, k)   源端质量
    b: (l,)         目标端质量 (均匀)
    M: (batch, k, l) 成本矩阵
    返回:
    T: (batch, k, l)
    """
    batch, k, l = M.shape
    device = M.device
    u = torch.ones((batch, k), device=device) / k
    v = torch.ones((batch, l), device=device) / l
    K = torch.exp(-M / epsilon)  # kernel
    for _ in range(max_iter):
        Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9
        u = a / Kv
        KTu = torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-9
        v = b / KTu
    T = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return T  # (batch, k, l)


# ================= 数据集 =================
class RealLiveStreamDataset(Dataset):
    def __init__(self, filepath, paid_cols, response_cols, context_cols=None, scaler_dict=None):
        """
        filepath: CSV 路径（分钟级数据）
        paid_cols: 干预（付费投流来源）列名
        response_cols: 响应变量列
        context_cols: 可选上下文列
        scaler_dict: {'X': StandardScaler, 'paid': StandardScaler} 可复用
        """
        df = pd.read_csv(filepath, encoding="utf-8")

        # 时间解析与排序（若存在“分钟”列）
        if "分钟" in df.columns:
            df["minute_ts"] = pd.to_datetime(df["分钟"], format="%H:%M", errors="coerce")
            df = df.sort_values("minute_ts").reset_index(drop=True)

        df = df.fillna(0.0)

        # 付费投流合并为一维（T,1）
        self.paid_raw = df[paid_cols].sum(axis=1).astype(float).values.reshape(-1, 1)
        # 响应向量（T, D）
        self.X_raw = df[response_cols].astype(float).values

        # 上下文（可选）
        self.context_raw = df[context_cols].astype(float).values if context_cols else None

        # 标准化：开源时建议仅用于训练过程，保存模型时默认不保存scaler
        if scaler_dict is None:
            self.scaler_X = StandardScaler()
            self.scaler_paid = StandardScaler()
            self.X = self.scaler_X.fit_transform(self.X_raw)
            self.paid = self.scaler_paid.fit_transform(self.paid_raw)
        else:
            self.scaler_X = scaler_dict["X"]
            self.scaler_paid = scaler_dict["paid"]
            self.X = self.scaler_X.transform(self.X_raw)
            self.paid = self.scaler_paid.transform(self.paid_raw)

        self.X = torch.from_numpy(self.X).float()            # (T, D)
        self.paid = torch.from_numpy(self.paid).float()      # (T, 1)
        self.context = torch.from_numpy(self.context_raw).float() if self.context_raw is not None else None

        # 无监督占位标签
        self.labels = torch.zeros(len(self.X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 关键修复：返回 (1,) 而不是 (1,1)，DataLoader 后为 (B,1)
        item = {
            "paid_flow": self.paid[idx],
            "X": self.X[idx],
            "label": self.labels[idx]
        }
        if self.context is not None:
            item["context"] = self.context[idx]
        return item


# ================= 模型组件 =================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[64, 64], activation=nn.ReLU):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HierarchicalInterventionVAE(nn.Module):
    def __init__(self, x_dim=2, z_base_dim=8, z_treat_dim=8,
                 theta0_dim=16, theta1_dim=16, num_prototypes=8, prototype_dim=16,
                 use_sinkhorn=True):
        super().__init__()
        self.z_base_dim = z_base_dim
        self.z_treat_dim = z_treat_dim
        self.theta0_dim = theta0_dim
        self.theta1_dim = theta1_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.use_sinkhorn = use_sinkhorn

        # encoders
        self.base_encoder = MLP(x_dim, z_base_dim * 2, hidden=[64, 64])
        self.treat_encoder = MLP(1 + z_base_dim, z_treat_dim * 2, hidden=[64, 64])

        # combine
        self.combine = MLP(z_base_dim + z_treat_dim, theta0_dim, hidden=[64])

        # prototypes + gate
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim) * 0.1)
        self.prototype_gate = MLP(1, num_prototypes, hidden=[32])

        # prior & decoder
        self.prior_theta1 = MLP(theta0_dim, theta1_dim * 2, hidden=[64])
        self.decoder = MLP(theta1_dim, x_dim, hidden=[64, 64])

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _map_theta0(self, t0):
        # map/crop to prototype_dim for prototype matching
        if self.theta0_dim == self.prototype_dim:
            return t0
        if self.theta0_dim < self.prototype_dim:
            return F.pad(t0, (0, self.prototype_dim - self.theta0_dim))
        return t0[:, : self.prototype_dim]

    def _prototype_weights(self, theta0_mapped, paid_flow):
        """
        计算样本对 K 原型的权重：
        - 若 use_sinkhorn: 用熵正则 OT（近似），强调理论意义；
        - 否则：softmax(-cost/ε) 更稳定。
        """
        B, K = theta0_mapped.size(0), self.num_prototypes

        # pairwise cost (B,K): 欧式距离平方
        diff = theta0_mapped.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B,K,D)
        cost = (diff ** 2).sum(dim=-1)  # (B,K)

        if not self.use_sinkhorn:
            # 退化情形：Boltzmann 分布（热度 ε）
            transport_weights = F.softmax(-cost / 0.05, dim=-1)  # (B,K)
        else:
            # --- 熵正则OT近似 ---
            # 这里将“样本->原型”作为单边分配近似来做：
            #   M: (B, K_src, K_tgt) 近似为 (B, K, K)，用 cost 列复制得到
            #   a: (B,K) 源端均匀，b: (K,) 目标端均匀
            # 说明：严格的双边配平需要构造真正的 K_src×K_tgt 成本，
            #       这里为了保持与论文叙述一致及实现简洁，使用该近似。
            a = torch.ones((B, K), device=theta0_mapped.device) / K
            b = torch.ones((K,), device=theta0_mapped.device) / K
            M = cost.unsqueeze(1).expand(B, K, K)  # (B,K,K)
            T = sinkhorn_transport(a, b, M, epsilon=0.05, max_iter=40)
            transport_weights = T.sum(dim=1)  # (B,K) 将运输计划在源端求边缘作为权重

        # 门控：基于 paid_flow 的 softmax gate
        gate = F.softmax(self.prototype_gate(paid_flow), dim=-1)  # (B,K)

        # 融合（元素乘）
        combined = transport_weights * gate
        combined = combined / (combined.sum(dim=-1, keepdim=True) + 1e-9)
        return combined, cost

    def _decode_path(self, theta0_vec):
        stats = self.prior_theta1(theta0_vec)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        theta1 = self.reparameterize(mu, logvar)
        recon_X = self.decoder(theta1)
        return recon_X, mu, logvar

    def forward(self, paid_flow, X):
        """
        paid_flow: (B,1) ; X: (B,D)
        """
        # 1) q(z_base|X)
        base_stats = self.base_encoder(X)
        mu_base, logvar_base = torch.chunk(base_stats, 2, dim=-1)
        z_base = self.reparameterize(mu_base, logvar_base)

        # 2) q(z_treat|paid_flow, z_base)
        treat_in = torch.cat([paid_flow, z_base], dim=-1)
        treat_stats = self.treat_encoder(treat_in)
        mu_treat, logvar_treat = torch.chunk(treat_stats, 2, dim=-1)
        z_treat = self.reparameterize(mu_treat, logvar_treat)

        # 3) 原路径：组合->原型对齐->prior->解码
        theta0_raw = self.combine(torch.cat([z_base, z_treat], dim=-1))
        theta0_map = self._map_theta0(theta0_raw)
        weights, cost = self._prototype_weights(theta0_map, paid_flow)
        theta0_prime = torch.matmul(weights, self.prototypes)  # (B, Dp)
        # 对齐到 theta0_dim
        if self.prototype_dim != self.theta0_dim:
            theta0_prime = theta0_prime[:, : self.theta0_dim] if self.prototype_dim > self.theta0_dim \
                           else F.pad(theta0_prime, (0, self.theta0_dim - self.prototype_dim))
        recon_X, mu_theta1, logvar_theta1 = self._decode_path(theta0_prime)

        # 4) 反事实路径：屏蔽 z_treat，且用“零投流”做门控
        theta0_cf_raw = self.combine(torch.cat([z_base, torch.zeros_like(z_treat)], dim=-1))
        theta0_cf_map = self._map_theta0(theta0_cf_raw)
        zero_paid = torch.zeros_like(paid_flow)
        weights_cf, cost_cf = self._prototype_weights(theta0_cf_map, zero_paid)
        theta0_cf_prime = torch.matmul(weights_cf, self.prototypes)
        if self.prototype_dim != self.theta0_dim:
            theta0_cf_prime = theta0_cf_prime[:, : self.theta0_dim] if self.prototype_dim > self.theta0_dim \
                              else F.pad(theta0_cf_prime, (0, self.theta0_dim - self.prototype_dim))
        recon_X_cf, mu_theta1_cf, logvar_theta1_cf = self._decode_path(theta0_cf_prime)

        # 5) KL
        kl_base = -0.5 * torch.sum(1 + logvar_base - mu_base.pow(2) - logvar_base.exp(), dim=1)
        kl_treat = -0.5 * torch.sum(1 + logvar_treat - mu_treat.pow(2) - logvar_treat.exp(), dim=1)
        kl_theta1 = -0.5 * torch.sum(1 + logvar_theta1 - mu_theta1.pow(2) - logvar_theta1.exp(), dim=1)

        # 6) 期望 prototype 代价
        expected_cost = torch.sum(weights * cost, dim=1)
        expected_cost_cf = torch.sum(weights_cf * cost_cf, dim=1)

        metrics = {
            "kl_base": kl_base,
            "kl_treat": kl_treat,
            "kl_theta1": kl_theta1,
            "pot_cost": expected_cost,
            "pot_cost_cf": expected_cost_cf,
            "z_base": z_base,
            "z_treat": z_treat
        }
        return recon_X, recon_X_cf, metrics


# ================= 损失与异常分数 =================
def loss_function(recon_X, recon_X_cf, X, metrics, beta_cf=1.0, beta_pot=1.0):
    recon_loss = F.mse_loss(recon_X, X, reduction="none").mean(dim=1)
    cf_gap = F.mse_loss(recon_X, recon_X_cf, reduction="none").mean(dim=1)
    kl_base = metrics["kl_base"]
    kl_treat = metrics["kl_treat"]
    kl_theta1 = metrics["kl_theta1"]
    pot_cost = metrics["pot_cost"]
    total = recon_loss + 0.1 * (kl_base + kl_treat + kl_theta1) + beta_pot * pot_cost + beta_cf * cf_gap
    return total.mean(), {
        "recon_loss": recon_loss.mean().item(),
        "cf_gap": cf_gap.mean().item(),
        "kl_base": kl_base.mean().item(),
        "kl_treat": kl_treat.mean().item(),
        "kl_theta1": kl_theta1.mean().item(),
        "pot_cost": pot_cost.mean().item()
    }


def anomaly_score_calculation(recon_X, recon_X_cf, X, metrics):
    recon_err = F.mse_loss(recon_X, X, reduction="none").mean(dim=1)
    cf_gap = F.mse_loss(recon_X, recon_X_cf, reduction="none").mean(dim=1)
    pot_cost = metrics["pot_cost"]
    score = recon_err + 0.5 * cf_gap + 0.2 * pot_cost
    return score.detach().cpu().numpy(), {
        "recon_err": recon_err.detach().cpu().numpy(),
        "cf_gap": cf_gap.detach().cpu().numpy(),
        "pot_cost": pot_cost.detach().cpu().numpy()
    }


# ================= 主流程 =================
def main():
    dataset = RealLiveStreamDataset(CSV_PATH, PAID_COLS, RESPONSE_COLS, CONTEXT_COLS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalInterventionVAE(
        x_dim=len(RESPONSE_COLS),
        use_sinkhorn=USE_SINKHORN
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader:
            paid_flow = batch["paid_flow"].to(device)  # (B,1)
            X = batch["X"].to(device)                  # (B,D)
            recon_X, recon_X_cf, metrics = model(paid_flow, X)
            loss, info = loss_function(recon_X, recon_X_cf, X, metrics, beta_cf=1.0, beta_pot=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f} "
              f"recon={info['recon_loss']:.4f} cf_gap={info['cf_gap']:.4f} pot={info['pot_cost']:.4f}")

    # 推断 + 可视化
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        paid_flow = batch["paid_flow"].to(device)
        X = batch["X"].to(device)
        recon_X, recon_X_cf, metrics = model(paid_flow, X)
        scores, _ = anomaly_score_calculation(recon_X, recon_X_cf, X, metrics)

    plt.figure(figsize=(7, 4))
    plt.scatter(range(len(scores)), scores, s=12, cmap="coolwarm")
    plt.title("异常分数 (示例)")
    plt.xlabel("样本索引")
    plt.ylabel("score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 仅保存权重（默认不保存 scaler）
    save_payload = {"model_state": model.state_dict()}
    if SAVE_SCALERS:
        # 警告：保存 scaler 可能反映真实数据分布统计，开源不建议
        save_payload.update({
            "scaler_X": dataset.scaler_X,
            "scaler_paid": dataset.scaler_paid
        })
    torch.save(save_payload, "intervention_vae_prototype.pth")
    print("模型已保存至 intervention_vae_prototype.pth",
          "(含 scaler)" if SAVE_SCALERS else "(不含 scaler)")


if __name__ == "__main__":
    main()
