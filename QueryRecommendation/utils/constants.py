import torch
MAX_SEMANTIC_LEN=25
MAX_STRUCT_FEATURE_LEN=16
SEMANTIC_POS={
    "NONE":0,
    "subspace-field":1,
    "subspace-value":2,
    "time-value":3,
    "time-type":4,
    "time-role":5,
    "measure-field":6,
    "measure-aggregation":7,
    "measure-type":8,
    "focus-field":9,
    "focus-value":10,
    "focus-level":11,
}
WORD_VECTORS_LEN=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')