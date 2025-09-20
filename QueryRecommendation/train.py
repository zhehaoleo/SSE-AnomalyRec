# -*- coding: utf-8 -*-
import os
import json
import time
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils.text_encoder_bge import BGETextEncoder
from utils.data_loader import pad_w2v_convert_fact_to_input_with_event_seq
from encoder.modeling import QueryRecommendationModel
from encoder.ActionAwareLoss import ActionAwareLoss
from utils.constants import *
from utils.jaccard import jaccard_rows

# =======================
# 全局配置
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", "path/to/bge-base-zh-v1.5")

# 数据与产物相对目录
DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
TB_DIR = os.getenv("TB_DIR", "runs")

# 训练超参
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-2
WEIGHT_DECAY = 1e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============ Text Encoder ============
text_encoder = BGETextEncoder(TEXT_MODEL_PATH, device=DEVICE, out_dim=128, normalize=True)


def _extract_sentences_from_item(item: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
    """从一个四元组样本中取出 sentence1..4；若缺失，用 {} 占位。"""
    s1 = item.get("sentence1", {}) or {}
    s2 = item.get("sentence2", {}) or {}
    s3 = item.get("sentence3", {}) or {}
    s4 = item.get("sentence4", {}) or {}
    return s1, s2, s3, s4


# =========================
# QA 四元组 batch 生成器
# =========================
def quadruplet_batch_generator_qa(
    all_data: List[Dict],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None
):
    """
    返回四个“原始 sentence 字典列表”：
    - x1_list = [sentence1, ...]
    - x2_list = [sentence2, ...]
    - x3_list = [sentence3, ...]
    - y1_list = [sentence4, ...]
    后续再用 pad_w2v_convert_fact_to_input_with_event_seq 做解析与 pad。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    data = list(all_data)
    data_size = len(data)
    idx = np.arange(data_size)

    while True:
        if shuffle:
            rng.shuffle(idx)

        start = 0
        while start < data_size:
            end = start + batch_size
            if end > data_size:
                if drop_last:  # 丢弃最后不满 batch 的一段
                    start = 0
                    break
                end = data_size

            batch_idx = idx[start:end]
            start = end

            x1_list, x2_list, x3_list, y1_list = [], [], [], []
            for i in batch_idx:
                s1, s2, s3, s4 = _extract_sentences_from_item(data[i])
                x1_list.append(s1)
                x2_list.append(s2)
                x3_list.append(s3)
                y1_list.append(s4)

            yield x1_list, x2_list, x3_list, y1_list


# =========================
# 把 loader 输出喂给模型
# =========================
def _to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def convert_raw_list_to_batch_inputs(batch_items: List[Dict]) -> Dict[str, np.ndarray]:
    return pad_w2v_convert_fact_to_input_with_event_seq(
        items=batch_items,
        method="decay+bucket",
        lambdas=(1, 3, 6, 12),
        bins=(0, 1, 2, 3, 5, 8, 13, 21, 34),
        d_bin=8,
        use_next=True,
        text_encoder=text_encoder,  # 与上方 text_encoder 一致
        D_feat=128                  # 与 text_encoder.out_dim 一致
    )


def get_embeddings_from_batch_dict(
    model: QueryRecommendationModel,
    batch_dict: Dict[str, np.ndarray]
) -> torch.Tensor:
    struct_one_hot = _to_tensor(batch_dict["batch_struct_one_hot"], torch.float32)
    indexed_tokens = batch_dict["batch_indexed_tokens"]
    pos_feat = _to_tensor(batch_dict["batch_pos"], torch.float32)

    evt_seq = batch_dict.get("batch_event_track_seq", None)
    evt_mask = batch_dict.get("batch_event_mask", None)

    if evt_seq is not None and hasattr(evt_seq, "size") and evt_seq.size and evt_seq.shape[-1] > 0:
        event_track_seq = _to_tensor(evt_seq, torch.float32)
        event_mask = _to_tensor(evt_mask, torch.float32)
        return model(struct_one_hot, indexed_tokens, pos_feat,
                     event_track_seq=event_track_seq,
                     event_mask=event_mask)
    else:
        return model(struct_one_hot, indexed_tokens, pos_feat)


# =========================
# 训练主流程
# =========================
def train(model_save_name: str, training_data_file_name: str):
    """
    model_save_name: 模型名（用于产物目录）
    training_data_file_name: 数据文件名（相对 data/ 目录）
    """
    # data
    training_data_file = os.path.join(DATA_DIR, training_data_file_name)
    if not os.path.exists(training_data_file):
        raise FileNotFoundError(
            f"[ERROR] 训练数据不存在：{training_data_file}\n"
            f"请把你的数据放在 {DATA_DIR}/ 下，文件名与代码一致，或通过环境变量 DATA_DIR 修改。"
        )

    model_subdir = "models"

    # 根输出目录（相对）
    base_out_dir = os.path.join(OUTPUT_DIR, "recommendation2vec")
    os.makedirs(base_out_dir, exist_ok=True)

    # 时间戳目录
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    training_model_folder_path = os.path.join(base_out_dir, model_subdir, f"{model_save_name}-{stamp}")
    os.makedirs(training_model_folder_path, exist_ok=True)

    # TensorBoard 目录
    tb_base = os.path.join(TB_DIR, "recommendation2vec")
    tensorboard_folder_path = os.path.join(tb_base, f"{model_save_name}-{stamp}")
    os.makedirs(tensorboard_folder_path, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_folder_path)

    # ---------- 加载数据 ----------
    with open(training_data_file, "r", encoding="utf-8") as f:
        all_train_data = json.load(f)
        # 过滤 comment 字段
        train_data = [
            {
                k: {kk: vv for kk, vv in v.items() if kk != "comment"} if isinstance(v, dict) else v
                for k, v in d.items()
            }
            for d in all_train_data
        ]

    batch_gen = quadruplet_batch_generator_qa(train_data, batch_size=BATCH_SIZE, seed=SEED)

    # 先窥探一批，确定事件 D_event 维度，决定是否启用事件分支
    peek_x1, _, _, _ = next(batch_gen)
    peek_bd_x1 = convert_raw_list_to_batch_inputs(peek_x1)
    if "batch_event_track_seq" in peek_bd_x1 and hasattr(peek_bd_x1["batch_event_track_seq"], "size") \
            and peek_bd_x1["batch_event_track_seq"].size > 0:
        event_in_dim = int(peek_bd_x1["batch_event_track_seq"].shape[-1])
    else:
        event_in_dim = 0

    # 重新创建生成器，避免丢第一批
    batch_gen = quadruplet_batch_generator_qa(train_data, batch_size=BATCH_SIZE, seed=SEED + 1)

    # ---------- 模型与损失 ----------
    qa_emb = QueryRecommendationModel(
        in_size=690,
        rep_size=10 * MAX_SEMANTIC_LEN + 400,
        hidden_dropout_prob=0.1,
        last_rep_size=300,
        event_in_dim=event_in_dim,
        event_hidden=128,
        event_out_dim=128,
        attn_heads=4, w_left=3, w_right=0
    ).to(DEVICE)

    loss_fc = ActionAwareLoss(alpha=0.25, margin=5, beta=1.0, triplet_mode="same_action_stronger").to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, qa_emb.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    steps_per_epoch = max(1, len(train_data) // BATCH_SIZE)
    global_step = 0
    best_loss = float("inf")

    # ---------- 训练 ----------
    for epoch in tqdm(range(EPOCHS), desc="epochs"):
        qa_emb.train()
        total_loss = 0.0

        for _ in tqdm(range(steps_per_epoch), desc="batch steps", leave=False):
            batch_x1, batch_x2, batch_x3, batch_y1 = next(batch_gen)

            bd_x1 = convert_raw_list_to_batch_inputs(batch_x1)
            bd_x2 = convert_raw_list_to_batch_inputs(batch_x2)
            bd_x3 = convert_raw_list_to_batch_inputs(batch_x3)
            bd_y1 = convert_raw_list_to_batch_inputs(batch_y1)

            res_x1 = get_embeddings_from_batch_dict(qa_emb, bd_x1)
            res_x2 = get_embeddings_from_batch_dict(qa_emb, bd_x2)
            res_x3 = get_embeddings_from_batch_dict(qa_emb, bd_x3)
            res_y1 = get_embeddings_from_batch_dict(qa_emb, bd_y1)

            # === 计算三连问的一致性权重 s ∈ [0,1] ===
            sig_a = bd_x1.get("action_signature", None)  # (B, E)
            sig_b = bd_x2.get("action_signature", None)  # (B, E)
            sig_c = bd_x3.get("action_signature", None)  # (B, E)

            if (sig_a is not None and sig_b is not None and sig_c is not None
                and hasattr(sig_a, "size") and hasattr(sig_b, "size") and hasattr(sig_c, "size")
                and sig_a.size and sig_b.size and sig_c.size):
                s1 = jaccard_rows(sig_a, sig_b)
                s2 = jaccard_rows(sig_b, sig_c)
                s_np = np.minimum(s1, s2).astype(np.float32)  # (B,)
                # 可选阈值：更严格地判定“同动作”可在此阈值化
                # s_np[s_np < 0.3] = 0.0
            else:
                # 无动作签名：可选 0（仅靠 triplet）或 1（退化为原论文）
                s_np = np.zeros((res_x1.size(0),), dtype=np.float32)

            s = torch.as_tensor(s_np, dtype=torch.float32, device=DEVICE)

            qa_loss = loss_fc(res_x1, res_x2, res_x3, res_y1, s=s)
            total_loss += float(qa_loss.detach().cpu().item())

            optimizer.zero_grad(set_to_none=True)
            qa_loss.backward()
            optimizer.step()

            tensorboard_writer.add_scalar("train/mean_loss", qa_loss.item(), global_step)
            global_step += 1

        # 记录 epoch 级损失
        avg_epoch_loss = total_loss / steps_per_epoch
        tensorboard_writer.add_scalar("train/epoch_loss_total", total_loss, epoch)
        tensorboard_writer.add_scalar("train/epoch_loss_avg", avg_epoch_loss, epoch)

        print(f"[epoch {epoch}] epoch_loss(total)={total_loss:.6f} "
              f"epoch_loss(avg)={avg_epoch_loss:.6f} best={best_loss:.6f}")

        scheduler.step()

        # 保存最优
        if total_loss < best_loss:
            best_loss = total_loss
            stamp2 = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            folder_path = os.path.join(training_model_folder_path, f"{stamp2}-ep{epoch}")
            os.makedirs(folder_path, exist_ok=True)
            torch.save({"model": qa_emb.state_dict()},
                       os.path.join(folder_path, "query_recommendation_model.pth"))

    print(f"[done] best_epoch_loss_total={best_loss:.6f} | saved under: {training_model_folder_path}")


if __name__ == "__main__":
    train(
        model_save_name="query_recommendation_model",
        training_data_file_name="dataset_with_event_track_v3_additive.json"
    )
