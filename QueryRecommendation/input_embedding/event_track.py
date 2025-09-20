# input_embedding/event_track.py
# -*- coding: utf-8 -*-
import numpy as np

class ExpDecayEncoder:
    """
    对稀疏事件脉冲 M[t,e] 做指数递推：
      s <- alpha * s + pulse
    输出多尺度拼接 [T, E*K]，等价于 time-since-last 的平滑表示。
    """
    def __init__(self, lambdas=(1,3,6,12)):
        self.lambdas = np.array(lambdas, dtype=np.float32)
        self.alphas = np.exp(-1.0 / self.lambdas)  # 每步的衰减

    def transform(self, M):
        T, E = M.shape
        K = len(self.lambdas)
        s = np.zeros((E, K), dtype=np.float32)
        out = np.zeros((T, E * K), dtype=np.float32)
        oneK = np.ones((1, K), dtype=np.float32)
        for t in range(T):
            # 关键改动：在最后一维做逐通道衰减
            s *= self.alphas[np.newaxis, :]  # 或者 s = s * self.alphas 也行

            # 脉冲注入：外积把每个事件通道扩到 K 个尺度
            s += M[t:t + 1, :].T @ oneK  # (E,1) @ (1,K) -> (E,K)

            out[t] = s.reshape(-1)
        return out  # [T, E*K]


class GapBucketEncoder:
    """
    对 gap_last / gap_next 做分桶 + 查表，得到 [T, E*(d 或 2d)] 的离散时间间隔表示。
    作为 Δt 编码（稀疏场景友好）。
    """
    def __init__(self, bins=(0,1,2,3,5,8,13,21,34), d_bin=8, use_next=True, max_gap=10**9, seed=42):
        self.bins = np.array(list(bins) + [10**9])
        self.B = len(self.bins)
        self.d_bin = d_bin
        self.use_next = use_next
        self.max_gap = max_gap
        rng = np.random.default_rng(seed)
        self.emb_last = rng.normal(0, 0.02, size=(self.B, d_bin)).astype(np.float32)
        self.emb_next = rng.normal(0, 0.02, size=(self.B, d_bin)).astype(np.float32) if use_next else None

    def _gap_scan(self, M, reverse=False):
        T, E = M.shape
        gaps = np.full((E,), self.max_gap, dtype=np.int64)
        out = np.zeros((T, E), dtype=np.int64)
        it = range(T-1, -1, -1) if reverse else range(T)
        for _, t in enumerate(it):
            gaps = np.where(M[t] > 0, 0, np.minimum(gaps + 1, self.max_gap))
            out[t] = gaps
        return out

    def _bucketize(self, gaps):
        ids = np.searchsorted(self.bins, gaps, side='left') - 1
        ids = np.clip(ids, 0, self.B - 1)
        return ids

    def transform(self, M):
        T, E = M.shape
        gap_last = self._gap_scan(M, reverse=False)
        ids_last = self._bucketize(gap_last)
        feat_last = self.emb_last[ids_last]  # [T,E,d]

        if self.use_next:
            gap_next = self._gap_scan(M, reverse=True)
            ids_next = self._bucketize(gap_next)
            feat_next = self.emb_next[ids_next]  # [T,E,d]
            feat = np.concatenate([feat_last, feat_next], axis=-1)  # [T,E,2d]
        else:
            feat = feat_last  # [T,E,d]

        return feat.reshape(T, -1).astype(np.float32)  # [T, E*(d or 2d)]


def build_event_track_features(
    M,
    method="decay+bucket",
    lambdas=(1,3,6,12),
    bins=(0,1,2,3,5,8,13,21,34),
    d_bin=8,
    use_next=True,
):
    """
    输入:
      M: [T, E] 稀疏事件脉冲/权重矩阵
    输出:
      feats: [T, D_event]  —— 不做池化！
    """
    feats = []
    if "decay" in method:
        feats.append(ExpDecayEncoder(lambdas=lambdas).transform(M))  # [T, E*K]
    if "bucket" in method:
        feats.append(GapBucketEncoder(bins=bins, d_bin=d_bin, use_next=use_next).transform(M))  # [T, E*(d or 2d)]
    if not feats:
        return np.zeros((M.shape[0], 0), dtype=np.float32)
    return np.concatenate(feats, axis=-1)  # [T, D_event]
