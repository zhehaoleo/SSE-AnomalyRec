import torch

def euclidean_similarity(x, y):
    """
    计算两个向量之间的欧几里得相似度。
    Inputs:
        x: list or torch tensor
        y: list or torch tensor
    Output:
        similarity: float, 越大越相似
    """
    # 强制转成Tensor
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y

    distance = torch.norm(x - y, p=2)  # 欧几里得距离
    similarity = 1 / (1 + distance)    # 变成越大越相似
    return similarity.item()


def batch_euclidean_similarity(batch_x, batch_y):
    """
    批量计算两个向量集合之间的欧几里得相似度
    Inputs:
        batch_x: shape (N, embedding_dim)
        batch_y: shape (M, embedding_dim)
    Output:
        sim_matrix: shape (N, M), 每个元素是 x[i] 和 y[j] 的相似度
    """
    sim_matrix = torch.zeros((batch_x.size(0), batch_y.size(0)))
    for i in range(batch_x.size(0)):
        for j in range(batch_y.size(0)):
            sim_matrix[i, j] = euclidean_similarity(batch_x[i], batch_y[j])
    return sim_matrix


# 示例
if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.5, 2.5, 3.5])
    sim = euclidean_similarity(a, b)
    print(f"欧几里得相似度: {sim:.4f}")