import numpy as np

def jaccard_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a,b: (B, E) 的 0/1 或 计数矩阵（计数会被 >0 当作 1）
    返回：逐样本的 Jaccard 相似度 (B,)
    """
    A = (a > 0).astype(np.float32)
    B = (b > 0).astype(np.float32)
    inter = np.sum(A * B, axis=1)
    union = np.sum((A + B) > 0, axis=1)
    out = np.zeros_like(inter, dtype=np.float32)
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out
