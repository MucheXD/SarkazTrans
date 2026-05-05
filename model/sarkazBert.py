from pathlib import Path
import torch.nn as nn
from transformers import BertModel, BertConfig

_BASE_DIR = Path(__file__).resolve().parent
_BERT_MODEL_DIR = _BASE_DIR / "bert-base-chinese"

bert_config = BertConfig.from_pretrained(str(_BERT_MODEL_DIR))

class SarkazBert(nn.Module):
    def __init__(self, bert_model: BertModel, dict_size=8100):
        super().__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.2)
        # 增加一层 LayerNorm 稳定切片后的特征分布
        self.norm = nn.LayerNorm(bert_model.config.hidden_size)
        self.mapper = nn.Linear(bert_model.config.hidden_size, dict_size)
        
        # 初始化：Xavier 初始化对 Linear 层通常效果更好
        nn.init.xavier_uniform_(self.mapper.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 考虑使用 last_hidden_state
        sequence_output = outputs.last_hidden_state
        
        # 执行切片操作
        sliced_output = sequence_output[:, 3:-1, :]
        
        # 流水线处理
        sliced_output = self.norm(sliced_output)
        sliced_output = self.dropout(sliced_output)
        
        return self.mapper(sliced_output)