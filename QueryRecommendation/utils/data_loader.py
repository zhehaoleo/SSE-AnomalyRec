import copy
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from input_embedding.structual_extract import extract_structual_info
from input_embedding.semantic_extract import extract_semantic_info
from utils.constants import *
from input_embedding.event_track import build_event_track_features
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

def w2v_convert_fact_to_input(item_fact_list):
    """
        Transform the fact type batch input for model input.
    """
    batch_indexed_tokens = list()
    batch_pos = list()
    batch_struct_one_hot = list()
    max_token_len = 0
    for item_fact in item_fact_list:
        tokenized_semantic_token, semantic_pos_list = extract_semantic_info(
            item_fact)
        strcuctual_one_hot = extract_structual_info(item_fact)
        if len(tokenized_semantic_token) > max_token_len:
            max_token_len = len(tokenized_semantic_token)
        batch_indexed_tokens.append(tokenized_semantic_token)
        batch_pos.append(semantic_pos_list)
        batch_struct_one_hot.append(strcuctual_one_hot)
    return {
        "batch_indexed_tokens": batch_indexed_tokens,
        "batch_pos": batch_pos,
        "batch_struct_one_hot": batch_struct_one_hot,
        "max_token_len": max_token_len
    }


def pad_w2v_convert_fact_to_input(item_fact_list):
    """
        Transform the fact type batch input for model input.
    """
    batch_indexed_tokens = list()
    batch_pos = list()
    batch_struct_one_hot = list()
    max_token_len = 0
    for item_fact in item_fact_list:
        tokenized_semantic_token, semantic_pos_list = extract_semantic_info(
            item_fact)
        strcuctual_one_hot = extract_structual_info(item_fact)
        if len(tokenized_semantic_token) > max_token_len:
            max_token_len = len(tokenized_semantic_token)
        batch_indexed_tokens.append(tokenized_semantic_token)
        batch_pos.append(semantic_pos_list)
        batch_struct_one_hot.append(strcuctual_one_hot)
    input = {
        "batch_indexed_tokens": batch_indexed_tokens,
        "batch_pos": batch_pos,
        "batch_struct_one_hot": batch_struct_one_hot,
    }
    return padding_one_batch_data_w2v(25, input)


def padding_one_batch_data_w2v(max_token_num, item_input):
    batch_indexed_tokens = item_input["batch_indexed_tokens"]
    new_batch_indexed_tokens = list()
    for item in batch_indexed_tokens:
        temp_index_token = copy.deepcopy(item)
        if len(item) < max_token_num:
            sub = max_token_num-len(item)
            for i in range(0, sub):
                temp_index_token.append("")
        else:
            temp_index_token = temp_index_token[0:max_token_num]
        new_batch_indexed_tokens.append(temp_index_token)

    batch_pos = item_input["batch_pos"]
    new_batch_pos = list()
    for item in batch_pos:
        temp_pos = copy.deepcopy(item)
        if len(item) < max_token_num:
            sub = max_token_num-len(item)
            for i in range(0, sub):
                temp_pos.append(0)
        else:
            temp_pos = temp_pos[0:max_token_num]
        new_batch_pos.append(temp_pos)
    return {
        "batch_indexed_tokens": new_batch_indexed_tokens,
        "batch_pos": np.array(new_batch_pos),
        "batch_struct_one_hot": np.array(item_input["batch_struct_one_hot"])
    }

# ------------------ 时间 & 工具 ------------------
_TZ_EAST8 = timezone(timedelta(hours=8))

def _to_fixed_dim(feat: np.ndarray, D_feat: int) -> np.ndarray:
    """
    把任意一维向量对齐到固定维度 D_feat：
      - 长度 >= D_feat: 截断
      - 长度 <  D_feat: 右侧 0 填充
    """
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    if feat.size >= D_feat:
        return feat[:D_feat]
    out = np.zeros((D_feat,), dtype=np.float32)
    out[:feat.size] = feat
    return out

# ------------------ 事件 → 特征向量（数值 or 文本） ------------------
def _event_feature_from_params(
    params: Dict[str, Any],
    text_encoder
) -> np.ndarray:
    """
    事件特征策略：
      - 数值优先：amount/value/face_value/coupon_value/duration_sec → [标量]
      - 若有 content（讲解文本）且提供了 text_encode_fn，则返回文本 embedding 向量
      - 否则 → [1.0]
    """
    if not params:
        return np.array([1.0], dtype=np.float32)

    # 数值类
    for k in ("amount", "value", "face_value", "coupon_value"):
        if k in params:
            try:
                return np.array([float(params[k])], dtype=np.float32)
            except Exception:
                pass
    if "duration_sec" in params:
        try:
            return np.array([float(params["duration_sec"])], dtype=np.float32)
        except Exception:
            pass

    # 文本讲解
    if "content" in params and text_encoder is not None:
        try:
            vec = text_encoder.encode(params["content"])  # np.ndarray [d_text]
            return vec.astype(np.float32)
        except Exception:
            pass

    return np.array([1.0], dtype=np.float32)

def _parse_iso(ts: str) -> float:
    ts = ts.replace(' ', 'T')
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ_EAST8)
    return dt.timestamp()

def _build_time_axis(start_iso: str, end_iso: str, bucket_sec: int) -> Tuple[float, float, int]:
    start_s = _parse_iso(start_iso)
    end_s   = _parse_iso(end_iso)
    assert end_s > start_s, f"time_window.end <= start: {end_iso} <= {start_iso}"
    T = int(np.ceil((end_s - start_s) / bucket_sec))
    return start_s, end_s, T

def _events_to_M_single_window(event_track: Dict[str, Any],
                               text_encoder=None,
                               D_feat: int = 128) -> Tuple[np.ndarray, List[str]]:
    """
    稀疏事件 → 脉冲矩阵（带特征通道展开）
    输出:
      - M_feat: [T, E*D_feat]
      - event_types: 事件类型顺序
    """
    tw = event_track.get("time_window", None)
    assert tw is not None and "start" in tw and "end" in tw, "event_track.time_window 缺少 start/end"
    bucket_sec = int(tw.get("bucket_sec", 60))
    start_s, end_s, T = _build_time_axis(tw["start"], tw["end"], bucket_sec)

    etypes = event_track.get("event_types", None)
    if etypes is None:
        etypes = sorted({ev.get("type", "UNKNOWN") for ev in event_track.get("events", [])})
    E = len(etypes)
    type2col = {t: i for i, t in enumerate(etypes)}

    M_feat = np.zeros((T, E * D_feat), dtype=np.float32)

    for ev in event_track.get("events", []):
        ts = ev.get("ts")
        tp = ev.get("type")
        if not ts or not tp or tp not in type2col:
            continue
        ts_s = _parse_iso(ts)
        if ts_s < start_s or ts_s >= end_s:
            continue
        t_idx = int((ts_s - start_s) // bucket_sec)
        if not (0 <= t_idx < T):
            continue

        e_idx = type2col[tp]
        params = ev.get("params", {}) or {}

        feat_vec = _event_feature_from_params(params, text_encoder=text_encoder)  # 可能是 [1] 或 [d_text]
        feat_vec = _to_fixed_dim(feat_vec, D_feat)                                # 对齐到 D_feat

        col_start = e_idx * D_feat
        col_end   = col_start + D_feat
        M_feat[t_idx, col_start:col_end] += feat_vec

    return M_feat.astype(np.float32), etypes

def _build_event_M_list_from_batch_items_sparse_only(
    batch_items: List[Dict],
    text_encoder=None,
    D_feat: int = 128,
) -> List[Optional[np.ndarray]]:
    Ms: List[Optional[np.ndarray]] = []
    for d in batch_items:
        et = d.get("event_track")
        if not et:
            Ms.append(None); continue
        try:
            M, _ = _events_to_M_single_window(et, text_encoder=text_encoder, D_feat=D_feat)
            Ms.append(M)   # [T, E*D_feat]
        except Exception as ex:
            print(f"[warn] 解析 event_track 失败，置 None：{ex}")
            Ms.append(None)
    return Ms

def _encode_event_seq_and_mask_from_sparse_M_list(
    event_M_list: List[Optional[np.ndarray]],
    method: str = "decay+bucket",
    lambdas=(1,3,6,12),
    bins=(0,1,2,3,5,8,13,21,34),
    d_bin=8,
    use_next=True,
    expected_event_dim_if_empty: int = 0,  # 新增：当整批为空时的期望维度（建议用全局配置算好传进来）
):
    """
    将一批 M（每个形状为 [T_i, channels]，也可能 None/空）编码为统一形状：
      batch_seq: [B, T_max, D_event]
      batch_msk: [B, T_max]
    规则：
      - 有样本非空 → 以第一条非空的 D_event 为准（也可校验一致性）
      - 样本空宽度或 None → 用零向量 (1, D_event) 占位，mask 的该步 = 1
      - 整批全空 → 若 expected_event_dim_if_empty>0，返回 [B,1,expected_dim] 全 0；否则返回 [B,1,0]
    """
    seqs = []           # 存每个样本编码后的 [T_i, D_i] 或 None/空
    lengths = []        # 每个样本的 T_i（空样本先记 1）
    D_event_ref = None  # 记录第一条非空序列的特征维度
    any_non_empty = False

    # 先逐样本编码，记录非空的 D_event
    for M in event_M_list:
        if M is None or M.size == 0:
            seqs.append(None)
            lengths.append(1)  # 空样本用 1 作为时间占位
            continue

        feat_seq = build_event_track_features(
            M, method=method, lambdas=lambdas, bins=bins, d_bin=d_bin, use_next=use_next
        )  # [T_i, D_i]

        if feat_seq.size > 0:
            if D_event_ref is None:
                D_event_ref = int(feat_seq.shape[-1])
            else:
                # 如果不一致，做个健壮性裁剪/填充；这里只做裁剪到较小值（也可以改成 pad）
                if feat_seq.shape[-1] != D_event_ref:
                    Dmin = min(D_event_ref, feat_seq.shape[-1])
                    if feat_seq.shape[-1] != Dmin:
                        feat_seq = feat_seq[:, :Dmin]
                    if D_event_ref != Dmin:
                        D_event_ref = Dmin
            seqs.append(feat_seq.astype(np.float32))
            lengths.append(int(feat_seq.shape[0]))
            any_non_empty = True
        else:
            seqs.append(None)
            lengths.append(1)

    # 决定最终 D
    if any_non_empty:
        D = D_event_ref if D_event_ref is not None else 0
    else:
        D = int(expected_event_dim_if_empty) if expected_event_dim_if_empty > 0 else 0

    # 决定 T_max
    T_max = max(lengths) if lengths else 1
    B = len(seqs)

    # 分配 batch 张量
    if D > 0:
        batch_seq = np.zeros((B, T_max, D), dtype=np.float32)
    else:
        # 真正无事件维度的退化形态（不建议长期使用；更推荐传 expected_event_dim_if_empty）
        batch_seq = np.zeros((B, 1, 0), dtype=np.float32)
        batch_msk = np.ones((B, 1), dtype=np.float32)  # 给个 1 占位
        return batch_seq, batch_msk

    batch_msk = np.zeros((B, T_max), dtype=np.float32)

    # 第二次遍历：填充
    for i, s in enumerate(seqs):
        if s is None or s.size == 0:
            # 用 (1, D) 的零向量占位
            batch_seq[i, 0, :] = 0.0
            batch_msk[i, 0] = 1.0
            continue

        # 若该样本的 D 与最终 D 不一致（上面已裁剪保证一致；这里再防御一次）
        if s.shape[-1] != D:
            if s.shape[-1] > D:
                s = s[:, :D]
            else:
                # 右侧 0-pad
                pad = np.zeros((s.shape[0], D - s.shape[-1]), dtype=np.float32)
                s = np.concatenate([s, pad], axis=-1)

        L = min(s.shape[0], T_max)
        batch_seq[i, :L, :] = s[:L, :]
        batch_msk[i, :L] = 1.0

    return batch_seq, batch_msk


# ------------------ 对外接口：with_event_seq ------------------
def pad_w2v_convert_fact_to_input_with_event_seq(
    items: List[Dict],
    event_M_list: Optional[List[Optional[np.ndarray]]] = None,
    method: str = "decay+bucket",
    lambdas=(1,3,6,12),
    bins=(0,1,2,3,5,8,13,21,34),
    d_bin=8,
    use_next=True,
    # 新增两个参数：
    text_encoder=None,   # 传入 BGETextEncoder 实例（可为 None）
    D_feat: int = 128,   # 每个事件类型展开的通道维度（用于序列特征）
    # 可选：动作签名是否用“计数”而不是 0/1
    use_count_signature: bool = False,
):
    """
    返回值在原有基础上新增：
      - base["action_signature"]: np.ndarray, 形状 (B, E_vocab)，0/1 或 计数
      - base["action_types_vocab"]: List[str], 事件类型词表（统一到整个 batch）
    """
    # ====== 1) 原有结构/语义处理（保持不变） ======
    base = pad_w2v_convert_fact_to_input(items)

    # ====== 2) 事件稀疏脉冲 -> 序列特征（保持你已有逻辑） ======
    if event_M_list is None:
        event_M_list = _build_event_M_list_from_batch_items_sparse_only(
            items, text_encoder=text_encoder, D_feat=D_feat
        )

    batch_seq, batch_msk = _encode_event_seq_and_mask_from_sparse_M_list(
        event_M_list, method=method, lambdas=lambdas, bins=bins, d_bin=d_bin, use_next=use_next
    )
    base["batch_event_track_seq"] = batch_seq   # [B, T_max, D_event]
    base["batch_event_mask"]      = batch_msk   # [B, T_max]

    # ====== 3) 新增：为每条 QA 计算“动作签名” ======
    # 目标：把每条样本的 QA 时间窗内发生过的事件类型做成 0/1（或计数）向量，词表对齐到整个 batch

    # 3.1 先为整个 batch 收集类型词表（优先使用 event_track.event_types；否则按出现排序）
    vocab_set = []
    for d in items:
        et = d.get("event_track", {})
        etypes = et.get("event_types", None)
        if etypes is None:
            # 从 events 中提取
            etypes = sorted({ev.get("type", "UNKNOWN") for ev in et.get("events", [])})
        vocab_set.extend(etypes)
    # 统一词表并固定顺序
    action_types_vocab = sorted(set(vocab_set))
    type2idx = {t: i for i, t in enumerate(action_types_vocab)}
    E_vocab = len(action_types_vocab)

    # 3.2 针对每条样本：在其 QA 时间窗内，统计每种类型是否出现（或出现次数）
    B = len(items)
    sig = np.zeros((B, E_vocab), dtype=np.float32)

    for b, d in enumerate(items):
        et = d.get("event_track", {})
        if not et:
            continue

        # 取该样本 QA 的时间窗
        tw = et.get("time_window", None)
        if tw is None or "start" not in tw or "end" not in tw:
            # 如果你的 QA 窗口不在 event_track 里，而在别的字段，请改这里
            continue
        start_s, end_s, _ = _build_time_axis(tw["start"], tw["end"], int(tw.get("bucket_sec", 60)))

        # 遍历事件，若在窗口内则记到签名向量
        for ev in et.get("events", []):
            tp = ev.get("type", None)
            ts = ev.get("ts", None)
            if tp is None or ts is None or tp not in type2idx:
                continue
            ts_s = _parse_iso(ts)
            if start_s <= ts_s <= end_s:
                j = type2idx[tp]
                if use_count_signature:
                    sig[b, j] += 1.0
                else:
                    sig[b, j] = 1.0  # 出现过即 1

    base["action_signature"] = sig                # (B, E_vocab)
    base["action_types_vocab"] = action_types_vocab

    return base

