# utils/text_encoder_bge.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Union, Optional

class BGETextEncoder:
    """
    本地 BGE 中文句向量编码器（默认加载 GPU）
    - model_path: 本地模型目录，如 "/data/models/bge-base-zh-v1.5"
    - device: "cuda" 或 "cpu"；默认自动检测
    - out_dim: 若设定，将线性降维到该维度（不设则保持模型默认维度 768）
    - normalize: 是否做 L2 归一化（检索/匹配更稳）
    """
    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None,
                 out_dim: Optional[int] = None,
                 normalize: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size  # 768 for bge-base-zh-v1.5

        self.normalize = normalize
        self.out_dim = out_dim
        if out_dim is not None and out_dim != self.hidden_size:
            self.proj = torch.nn.Linear(self.hidden_size, out_dim).to(self.device)
        else:
            self.proj = None

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        emb = self.encode_batch([text])  # [1, d]
        return emb[0]

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)  # last_hidden_state: [B, L, H]
            # mean pooling（也可改 CLS 或 attention mask 加权 mean）
            feats = outputs.last_hidden_state.mean(dim=1)  # [B, H]
            if self.proj is not None:
                feats = self.proj(feats)                   # [B, out_dim]
            if self.normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            vecs.append(feats.detach().cpu().numpy())
        return np.concatenate(vecs, axis=0)  # [N, D]
