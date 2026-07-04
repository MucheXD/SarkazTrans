from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SarkazBert(nn.Module):
    def __init__(self, mlm_model: Any, dict_size: int = 8100):
        super().__init__()
        self.dict_size = dict_size

        # Core-side compact token embedding. head_ids still use BERT's pretrained
        # word embeddings; core_ids use this 30-token space and are projected back
        # to BERT hidden size.
        self.embedding = nn.Embedding(num_embeddings=30, embedding_dim=768)
        self.reconstructor_mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 768),
            nn.Dropout(0.2),
        )
        self.reconstructor_norm = nn.LayerNorm(768)

        self.bert_model = mlm_model.bert
        self.mlm_head = mlm_model.cls
        self.drop_and_norm = nn.Sequential(
            nn.Dropout(0.2),
            nn.LayerNorm(self.bert_model.config.hidden_size),
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
    ) -> torch.Tensor:
        # Head: keep BERT's pretrained embedding for [CLS], type marker, [SEP].
        head_embeddings = self.bert_model.embeddings.word_embeddings(head_ids)

        # Core: learn compact-input reconstruction into BERT hidden space.
        core_embeddings = self.embedding(core_ids)
        core_transformed = self.reconstructor_mlp(core_embeddings)
        core_embeddings = self.reconstructor_norm(core_embeddings + core_transformed)

        embeddings = torch.cat([head_embeddings, core_embeddings], dim=1)
        bert_outputs = self.bert_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = bert_outputs.last_hidden_state[:, 3:, :]
        sequence_output = self.drop_and_norm(sequence_output)
        full_logits = self.mlm_head(sequence_output)
        logits = full_logits[:, :, : self.dict_size]
        if logits.size(-1) != self.dict_size:
            raise RuntimeError("MLM logits were not sliced to dict_size")
        return logits
