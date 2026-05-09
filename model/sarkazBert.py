from pathlib import Path
import torch.nn as nn
import torch
from transformers import BertForMaskedLM, BertConfig

_BASE_DIR = Path(__file__).resolve().parent
_BERT_MODEL_DIR = _BASE_DIR / "bert-base-chinese"

bert_config = BertConfig.from_pretrained(str(_BERT_MODEL_DIR))

class SarkazBert(nn.Module):
  def __init__(self, mlm_model: BertForMaskedLM, dict_size=8100):
    super().__init__()
    self.dict_size = dict_size
    
    # Layer1 - Reconstructor
    self.embedding = nn.Embedding(num_embeddings=30, embedding_dim=768)
    self.reconstructor_mlp = nn.Sequential(
      nn.Linear(768, 768),
      nn.GELU(),
      nn.Linear(768, 768),
      nn.Dropout(0.2)
    )
    self.reconstructor_norm = nn.LayerNorm(768)

    # Layer2 - BertModel 与 MLM head 都来自预训练权重
    self.bert_model = mlm_model.bert
    self.mlm_head = mlm_model.cls
    
    self.drop_and_norm = nn.Sequential(
      nn.Dropout(0.2),
      nn.LayerNorm(self.bert_model.config.hidden_size)
    )

  def set_bert_trainable(self, trainable: bool) -> None:
    for param in self.bert_model.parameters():
      param.requires_grad = trainable
    for param in self.mlm_head.parameters():
      param.requires_grad = trainable

  def forward(
    self,
    head_ids: torch.Tensor,
    core_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
  ):

    # head 部分继续使用 BERT 原始 embedding，保留 CLS / TYPE / SEP 的预训练表示
    head_embeddings = self.bert_model.embeddings.word_embeddings(head_ids)

    # core 部分使用自定义 embedding，并经过重建层变换
    core_embeddings = self.embedding(core_ids)
    core_transformed = self.reconstructor_mlp(core_embeddings)
    core_embeddings = self.reconstructor_norm(core_embeddings + core_transformed)

    # 拼成完整输入，供 BERT 编码器处理
    new_embeddings = torch.cat([head_embeddings, core_embeddings], dim=1)
    
    # 送入 BERT 核心层
    bert_outputs = self.bert_model(
        inputs_embeds=new_embeddings,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )

    # 使用 last_hidden_state
    sequence_output = bert_outputs.last_hidden_state
    
    # 执行切片操作（去除 head 部分，只保留 core 输出）
    sliced_output = sequence_output[:, 3:, :]
    
    # 对切片后的输出进行 Dropout 和 LayerNorm
    sliced_output = self.drop_and_norm(sliced_output)
    
    full_logits = self.mlm_head(sliced_output)
    logits = full_logits[:, :, : self.dict_size]
    assert logits.size(-1) == self.dict_size, "MLM logits were not sliced to dict_size"
    return logits