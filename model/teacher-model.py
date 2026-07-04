from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import BertForMaskedLM

from tokenizer import SarkazTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR.parent / "resources" / "bert-base-chinese"
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
PAD_ID = 0


def _pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_tensor(values: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(values):
        tensor = values.to(device=device, dtype=torch.long)
    else:
        tensor = torch.tensor(values, dtype=torch.long, device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError("target_ids and attention_mask must be 1D or 2D tensors")
    return tensor


def _special_token_ids(tokenizer: SarkazTokenizer) -> set[int]:
    return {
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.cls_id,
        tokenizer.sep_id,
        tokenizer.magic_id_0,
        tokenizer.magic_id_1,
    }


def _normalize_display_token(token: str | None) -> str:
    if token is None:
        return "[UNK]"
    return str(token).replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␠")


class TeacherModel:
    """BERT MLM teacher that scores each requested position by masking it first.

    `target_ids` and `attention_mask` describe the teacher-side text sequence without
    [CLS]/[SEP]. `token_mask` selects which non-padding positions should be predicted.
    This distinction is important: token_mask excludes non-output delimiters, while
    attention_mask excludes only padding.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | torch.device | None = None):
        self.device = _pick_device(str(device) if isinstance(device, torch.device) else device)
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = SarkazTokenizer()
        self.model = BertForMaskedLM.from_pretrained(str(self.model_dir), local_files_only=True)
        self.mask_token_id = getattr(self.model.config, "mask_token_id", MASK_ID) or MASK_ID
        self.cls_token_id = getattr(self.model.config, "cls_token_id", CLS_ID) or CLS_ID
        self.sep_token_id = getattr(self.model.config, "sep_token_id", SEP_ID) or SEP_ID
        self.pad_token_id = getattr(self.model.config, "pad_token_id", PAD_ID) or PAD_ID
        self.model.to(self.device)
        self.model.eval()

    def Answer(
        self,
        target_ids: Sequence[int] | torch.Tensor,
        attention_mask: Sequence[int] | torch.Tensor,
        token_mask: Sequence[int] | torch.Tensor | None = None,
        vector_chunk_size: int = 64,
    ) -> torch.Tensor:
        """Return per-position logits from repeated masked-token forward passes.

        Args:
            target_ids: Tensor/list with shape (batch, seq_len). These are BERT
                vocabulary ids for the teacher context.
            attention_mask: Same shape. 1 means the position is real context, 0 padding.
            token_mask: Optional same-shape mask. 1 means this position should be
                masked and predicted. If omitted, every attention_mask==1 position is
                predicted.
            vector_chunk_size: Number of masked sequences evaluated per teacher call.
                Lower it if a 12G GPU still OOMs on long sequences.

        Returns:
            Float32 tensor with shape (batch, seq_len, teacher_vocab_size). Positions
            not selected by token_mask are all zeros.
        """
        if vector_chunk_size <= 0:
            raise ValueError("vector_chunk_size must be positive")

        target_ids_tensor = _as_tensor(target_ids, self.device)
        attention_mask_tensor = _as_tensor(attention_mask, self.device)

        if target_ids_tensor.shape != attention_mask_tensor.shape:
            raise ValueError(
                f"target_ids shape {tuple(target_ids_tensor.shape)} must match attention_mask shape {tuple(attention_mask_tensor.shape)}"
            )

        if token_mask is None:
            prediction_mask = attention_mask_tensor.bool()
        else:
            token_mask_tensor = _as_tensor(token_mask, self.device)
            if token_mask_tensor.shape != target_ids_tensor.shape:
                raise ValueError(
                    f"token_mask shape {tuple(token_mask_tensor.shape)} must match target_ids shape {tuple(target_ids_tensor.shape)}"
                )
            prediction_mask = token_mask_tensor.bool() & attention_mask_tensor.bool()

        batch_size, seq_len = target_ids_tensor.shape
        attention_mask_bool = attention_mask_tensor.bool()

        wrapped_len = seq_len + 2
        wrapped_input_ids = torch.full(
            (batch_size, wrapped_len), self.pad_token_id, dtype=torch.long, device=self.device
        )
        wrapped_attention_mask = torch.zeros((batch_size, wrapped_len), dtype=torch.long, device=self.device)
        wrapped_token_type_ids = torch.zeros((batch_size, wrapped_len), dtype=torch.long, device=self.device)

        wrapped_input_ids[:, 0] = self.cls_token_id
        wrapped_input_ids[:, 1 : seq_len + 1] = target_ids_tensor
        wrapped_attention_mask[:, 0] = 1
        wrapped_attention_mask[:, 1 : seq_len + 1] = attention_mask_tensor.long()

        # Put SEP immediately after the last real token for each sample. This avoids
        # placing SEP after padding when the batch contains mixed lengths.
        lengths = attention_mask_tensor.long().sum(dim=1).clamp(min=0, max=seq_len)
        batch_arange = torch.arange(batch_size, device=self.device)
        sep_cols = lengths + 1
        wrapped_input_ids[batch_arange, sep_cols] = self.sep_token_id
        wrapped_attention_mask[batch_arange, sep_cols] = 1

        valid_indices = torch.nonzero(prediction_mask, as_tuple=False)
        num_masked = int(valid_indices.size(0))
        result = torch.zeros(
            (batch_size, seq_len, self.model.config.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        if num_masked == 0:
            return result

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = device_type == "cuda"
        with torch.inference_mode():
            with torch.autocast(device_type=device_type, enabled=use_amp):
                for start_idx in range(0, num_masked, vector_chunk_size):
                    end_idx = min(start_idx + vector_chunk_size, num_masked)
                    chunk_indices = valid_indices[start_idx:end_idx]
                    chunk_size = int(chunk_indices.size(0))

                    source_rows = chunk_indices[:, 0]
                    mask_cols = chunk_indices[:, 1] + 1  # shifted by inserted CLS
                    row_arange = torch.arange(chunk_size, device=self.device)

                    chunk_input_ids = wrapped_input_ids[source_rows].clone()
                    chunk_attention_mask = wrapped_attention_mask[source_rows]
                    chunk_token_type_ids = wrapped_token_type_ids[source_rows]
                    chunk_input_ids[row_arange, mask_cols] = self.mask_token_id

                    # Run only the encoder, then apply the MLM head only to the masked
                    # hidden states. This avoids materializing (chunk, seq, vocab) logits.
                    bert_outputs = self.model.bert(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        token_type_ids=chunk_token_type_ids,
                        return_dict=True,
                    )
                    masked_hidden = bert_outputs.last_hidden_state[row_arange, mask_cols]
                    chunk_logits = self.model.cls(masked_hidden.unsqueeze(1)).squeeze(1).to(dtype=result.dtype)
                    result[chunk_indices[:, 0], chunk_indices[:, 1]] = chunk_logits

        return result

    def unload(self) -> None:
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _encode_input(tokenizer: SarkazTokenizer, raw_text: str) -> torch.Tensor:
    normalized = tokenizer._normalize(raw_text)
    if not normalized:
        raise ValueError("input text cannot be empty")
    target_ids = tokenizer.encode_target(normalized)
    return torch.tensor([target_ids], dtype=torch.long)


def _build_attention_mask(target_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(target_ids, dtype=torch.long)


def _print_topk_table(
    tokenizer: SarkazTokenizer,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float,
    topk: int,
) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    special_ids = _special_token_ids(tokenizer)
    seq_len = logits.size(1)
    vocab_size = logits.size(-1)
    k = min(topk, vocab_size)

    print(f"Input: {tokenizer.decode(target_ids[0].tolist())}")
    for position in range(seq_len):
        position_logits = logits[0, position].clone()
        if special_ids:
            index = torch.tensor(sorted(special_ids), device=position_logits.device)
            position_logits[index] = float("-inf")

        probabilities = torch.softmax(position_logits / temperature, dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=k)
        source_token_id = int(target_ids[0, position].item())
        source_token = _normalize_display_token(tokenizer.id_to_token.get(source_token_id, "?"))

        cells = []
        for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            token = _normalize_display_token(tokenizer.id_to_token.get(token_id, "?"))
            cells.append(f"{token} {prob:.4f}")

        print(f"Pos {position + 1:02d} | {source_token} | " + " | ".join(cells) + " |")


def _iter_inputs() -> Iterable[str]:
    if sys.stdin.isatty():
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            if text.lower() in {"exit", "quit", "q"}:
                return
            yield text
        return

    for line in sys.stdin:
        text = line.strip()
        if text:
            yield text


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher model smoke test")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vector-chunk-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _get_args()
    teacher = TeacherModel(model_dir=args.model_dir, device=args.device)

    if sys.stdin.isatty():
        print("Enter text to tokenize and score. Type exit to quit.")

    for raw_text in _iter_inputs():
        try:
            target_ids = _encode_input(teacher.tokenizer, raw_text)
            attention_mask = _build_attention_mask(target_ids)
            logits = teacher.Answer(
                target_ids,
                attention_mask,
                token_mask=attention_mask,
                vector_chunk_size=args.vector_chunk_size,
            )
            _print_topk_table(teacher.tokenizer, logits, target_ids, args.temperature, args.topk)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
