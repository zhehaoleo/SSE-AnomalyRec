import torch
import torch.nn as nn
import numpy as np
from utils.constants import *
# from wikipedia2vec import Wikipedia2Vec
from sentence_transformers import SentenceTransformer
from encoder.attn_pool import AttnWindowPool

class SentenceEncoderVector(nn.Module):
    """
        Based on the fact token, use wikipedia2vec to get the word vector
        Inputs:
            `tokenized_text_list`: The words obtained by the tokenizer, which can be used to obtain the word vector later.
    """
    def __init__(self, model_path: str, window_size=10):
        super(SentenceEncoderVector, self).__init__()
        # location of the word vector model
        self.word2vec_model = SentenceTransformer(model_path)
        self.w_avg_pool=nn.AvgPool1d(window_size, stride=window_size)

    def forward(self, tokenized_text_list):
        word2vec_list=[]
        cache = {}
        for sentence in tokenized_text_list:
            temp_sentence_list=[]
            for word in sentence:
                try:
                    if word in cache:
                        word_vectors=cache[word]
                    else:
                        word_vectors=self.word2vec_model.encode(word.lower())
                        cache[word]=word_vectors
                except:
                    word_vectors=[0]*WORD_VECTORS_LEN
                temp_sentence_list.extend(np.array(word_vectors))
            word2vec_list.append(temp_sentence_list)
        temp_list=np.array(word2vec_list)
        temp_sentence_tensor=torch.FloatTensor(temp_list).to(device)
        # fuzzy processing of word vectors to extract thematic information
        output=self.w_avg_pool(temp_sentence_tensor)
        return output


class StructuralEmbedding(nn.Module):
    """
        According to the grammar tree, the structure of the fact is encoded, and CNN is used.
        Inputs:
            `item_struct`: One-hot matrices composed of structural information.
    """
    def __init__(self, rules_feature=MAX_STRUCT_FEATURE_LEN,out_channels=10,kernel_size=3):
        super(StructuralEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(rules_feature, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Flatten()
        )

    def forward(self, item_struct):
        return self.conv(item_struct)

class QueryRecommendationModel(nn.Module):
    """
    QueryRecommendation model, capturing information from both structural and semantic aspects,
    with an optional Event-Track branch using local attention pooling.

    Inputs:
        struct_one_hot: One-hot matrices composed of structural information.   [B, ...]
        semantic_tokens: Words extracted from semantic information.            [B, ...]
        semantic_pos: Position of semantically-informed words.                 [B, ...]
        (optional)
        event_track_seq: Event-track time sequence per sample.                 [B, T, event_in_dim]
        event_mask: Valid-time mask for each sample.                           [B, T] (1=valid, 0=pad)
    """
    def __init__(self,
                 in_size=690,
                 rep_size=10*128+400,  # 这里保留原来的写法（MAX_SEMANTIC_LEN=128时），你也可直接传参覆写
                 hidden_dropout_prob=0.3,
                 last_rep_size=512,
                 # ==== 新增：事件分支参数 ====
                 event_in_dim: int = 0,       # =0 表示关闭事件分支；>0 表示启用
                 event_hidden: int = 128,     # 事件序列先投到的维度（等同 H）
                 event_out_dim: int = 128,    # 聚合后的事件向量维度（会加到 fc 的输入维）
                 attn_heads: int = 4,
                 w_left: int = 3,
                 w_right: int = 0):
        super(QueryRecommendationModel, self).__init__()

        self.structual_embedding = StructuralEmbedding()
        self.semantic_embedding  = SemanticEmbedding()

        # ==== 事件分支（可选）====
        self.use_event = event_in_dim > 0
        if self.use_event:
            # 事件序列按步线性投影到 event_hidden
            self.event_in = nn.Linear(event_in_dim, event_hidden, bias=False)
            # 局部注意力池化（形态A）
            self.event_pool = AttnWindowPool(
                D_in=event_hidden,
                heads=attn_heads,
                W_left=w_left,
                W_right=w_right,
                out_dim=event_hidden,
                dropout=0.1
            )
            # 时间维聚合（mean）+ 门控 + 输出到 event_out_dim
            self.event_gate = nn.Sequential(nn.Linear(event_hidden, event_hidden), nn.Sigmoid())
            self.event_out  = nn.Linear(event_hidden, event_out_dim)

        # ==== 根据是否启用事件分支，自动确定 fc 第一层的输入维度 ====
        concat_in = in_size + (event_out_dim if self.use_event else 0)

        self.fc = nn.Sequential(
            nn.Linear(concat_in, rep_size),
            nn.BatchNorm1d(rep_size),
            nn.ReLU(inplace=True),
            nn.Linear(rep_size, last_rep_size)
        )
        self.dropout  = nn.Dropout(hidden_dropout_prob)
        self.batchNorm = nn.BatchNorm1d(last_rep_size)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor):
        """
        x:    [L, D]
        mask: [L]  (1/0)
        """
        if mask is None:
            return x.mean(dim=0)
        m = mask.float().unsqueeze(-1)              # [L,1]
        s = (x * m).sum(dim=0)                      # [D]
        d = m.sum().clamp_min(1.0)                  # [1]
        return s / d

    def forward(self,
                struct_one_hot,
                semantic_tokens,
                semantic_pos,
                event_track_seq = None,   # [B,T,event_in_dim]
                event_mask = None         # [B,T]
                ):
        # 结构/语义分支
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        struct_one_hot = torch.as_tensor(struct_one_hot, dtype=torch.float32, device=device)
        structual_embed = self.structual_embedding(struct_one_hot)                 # [B, D_struct]
        semantic_embed  = self.semantic_embedding(semantic_tokens, semantic_pos)   # [B, D_seman]

        concat_list = [structual_embed, semantic_embed]

        # 事件分支（可选）
        if self.use_event and event_track_seq is not None and event_track_seq.numel() > 0:
            B, T, Din = event_track_seq.shape
            evt_seq = event_track_seq.to(structual_embed.device)                   # [B,T,event_in_dim]
            msk_seq = (event_mask.to(structual_embed.device) if event_mask is not None
                       else torch.ones(B, T, device=evt_seq.device))

            pooled_batch = []
            for b in range(B):
                L = int(msk_seq[b].sum().item())
                if L <= 0:
                    pooled_batch.append(torch.zeros(self.event_out.out_features, device=evt_seq.device))
                    continue
                e_seq = evt_seq[b, :L, :]                 # [L, event_in_dim]
                e_seq = self.event_in(e_seq)              # [L, event_hidden]
                e_attn = self.event_pool(e_seq)           # [L, event_hidden]
                # 时间维聚合（mask-aware mean）
                h_event_pooled = self._masked_mean(e_attn, torch.ones(L, device=e_attn.device))  # [event_hidden]
                # 门控 + 输出
                h_event_pooled = self.event_out(self.event_gate(h_event_pooled) * h_event_pooled)      # [event_out_dim]
                pooled_batch.append(h_event_pooled)

            h_event = torch.stack(pooled_batch, dim=0)      # [B, event_out_dim]
            concat_list.append(h_event)

        # 融合 + 两层全连接（与原版保持一致）
        concate_structure_semantic_tensor = torch.cat(concat_list, dim=1)
        fusion_tensor = self.fc(concate_structure_semantic_tensor)
        fusion_tensor = self.batchNorm(fusion_tensor)
        fusion_tensor = self.dropout(fusion_tensor)
        return fusion_tensor


class SemanticEmbedding(nn.Module):
    """
        token：分词后的 list[list[str]]（喂给 Word2vecVector）
        pos  ：位置索引 [B, L]
    """
    def __init__(self, fact_pos_num=len(SEMANTIC_POS.keys()), hidden_size=10):
        super(SemanticEmbedding, self).__init__()
        # You should load the model path from a configuration or environment variable
        # For example: TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", "path/to/bge-base-zh-v1.5")
        self.w2v_embeddings = SentenceEncoderVector(model_path="path/to/your/model")
        self.fact_pos_embeddings = nn.Embedding(fact_pos_num, hidden_size)
        self.reduce_dim = nn.Linear(1920, 250)

    def forward(self, semantic_tokens, semantic_pos):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) 位置索引 → long + 正确的 device
        if torch.is_tensor(semantic_pos):
            fact_pos_id = semantic_pos.to(device=device, dtype=torch.long)
        else:
            fact_pos_id = torch.as_tensor(semantic_pos, dtype=torch.long, device=device)

        # 2) 位置嵌入
        pos_emb = self.fact_pos_embeddings(fact_pos_id)  # 已在同一 device，无需再 .to
        pos_emb = torch.reshape(pos_emb, (pos_emb.size(0), pos_emb.size(1) * pos_emb.size(2)))

        # 3) token 向量 → 确保在同一 device
        token_emb = self.w2v_embeddings(semantic_tokens)
        token_emb = token_emb.to(device)  # 若 Word2vecVector 返回在 CPU，需要搬到同一 device

        token_emb_reduce_dim_250 = self.reduce_dim(token_emb)

        semantic_emb = pos_emb + token_emb_reduce_dim_250
        return semantic_emb




