from pathlib import Path
from typing import Dict, List, Any

import torch


class SarkazTokenizer:
    """Project tokenizer and batch collator.

    Input side:
      - head_ids are always [CLS, type-token, SEP].
      - core_ids use a compact alphabet space: a-z => 1..26, [, ], | => 27..29, pad => 0.

    Target side:
      - target_ids are aligned to core_ids. Positions where token_mask == 0 carry a
        context placeholder id and are ignored by the loss.
      - token_mask marks real output-token positions. It is not a padding mask.
      - core_attention_mask marks non-padding core positions.
    """

    def __init__(self, vocab_file=None):
        base_dir = Path(__file__).resolve().parent
        default_vocab = base_dir / "data" / "vocab.txt"
        self.vocab_file = Path(vocab_file) if vocab_file else default_vocab

        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        with self.vocab_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        self.pad_id = 0
        self.unk_id = 100  # used only for target-side unknown characters
        self.cls_id = 101
        self.sep_id = 102
        self.magic_id_0 = 6432  # "说"
        self.magic_id_1 = 5031  # "答"

        # a-z: 1-26, [: 27, ]: 28, |: 29
        self._input_char_to_id = {
            "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
            "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
            "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
            "[": 27, "]": 28, "|": 29,
        }
        # These input-side symbols occupy a core position but do not correspond to an output token.
        self._special_target_tokens = {"[", "]", "|", ";"}

    def _normalize(self, text: str) -> str:
        return "".join(text.split()).lower()

    def _encode_core(self, text: str):
        normalized = self._normalize(text)
        input_core: List[int] = []
        token_mask: List[int] = []

        for ch in normalized:
            input_core.append(self._input_char_to_id.get(ch, self.pad_id))
            token_mask.append(0 if ch in self._special_target_tokens else 1)

        return input_core, token_mask

    def _build_token_type_ids(self, ids: List[int]) -> List[int]:
        type_ids: List[int] = []
        segment_id = 0
        for token_id in ids:
            type_ids.append(segment_id)
            if token_id == self.sep_id:
                segment_id = 1
        return type_ids

    def _target_context_id_for_masked_core_char(self, ch: str) -> int:
        """Return a BERT-vocab id used only as teacher context for token_mask==0 positions."""
        return self.token_to_id.get(ch, self.pad_id)

    def encode(self, t: int, i: str, o: str) -> Dict[str, List[int]]:
        if not isinstance(t, int):
            raise TypeError("t must be int")
        if not isinstance(i, str):
            raise TypeError("i must be str")
        if not isinstance(o, str):
            raise TypeError("o must be str")

        if t == 0:
            magic_id = self.magic_id_0
        elif t == 1:
            magic_id = self.magic_id_1
        else:
            raise ValueError("t must be 0 or 1")

        normalized_input = self._normalize(i)
        core_ids, token_mask = self._encode_core(i)
        compact_target_ids = self.encode_target(o)

        valid_token_count = sum(token_mask)
        if valid_token_count != len(compact_target_ids):
            raise ValueError(
                f"Valid token count ({valid_token_count}) must equal target_ids length ({len(compact_target_ids)}). "
                f"Input: '{i}' -> {len(core_ids)} core tokens, {valid_token_count} valid. "
                f"Output: '{o}' -> {len(compact_target_ids)} tokens."
            )

        # Align targets back to core positions. token_mask==0 positions are ignored by
        # loss, but carrying a readable context id lets the teacher keep delimiters in
        # its attention context during offline knowledge extraction.
        target_ids: List[int] = []
        target_cursor = 0
        for ch, is_output_position in zip(normalized_input, token_mask):
            if is_output_position:
                target_ids.append(compact_target_ids[target_cursor])
                target_cursor += 1
            else:
                target_ids.append(self._target_context_id_for_masked_core_char(ch))

        if target_cursor != len(compact_target_ids):
            raise AssertionError("target alignment cursor did not consume all compact targets")

        return {
            "head_ids": [self.cls_id, magic_id, self.sep_id],
            "core_ids": core_ids,
            "target_ids": target_ids,
            "token_mask": token_mask,
        }

    def encode_target(self, text: str) -> List[int]:
        """Encode target text one Unicode character at a time."""
        if not isinstance(text, str):
            raise TypeError("text must be str")

        normalized = self._normalize(text)
        return [self.token_to_id.get(ch, self.unk_id) for ch in normalized]

    def collate(self, batch) -> Dict[str, torch.Tensor]:
        if not isinstance(batch, list):
            raise TypeError("batch must be a list of encoded samples")
        if not batch:
            raise ValueError("batch must not be empty")

        head_batches: List[List[int]] = []
        core_batches: List[List[int]] = []
        target_batches: List[List[int]] = []
        token_mask_batches: List[List[int]] = []
        sample_indices: List[int] = []
        has_sample_index = all("sample_index" in item for item in batch)
        has_knowledge = any("teacher_topk_ids" in item or "teacher_topk_probs" in item for item in batch)
        if has_knowledge and not all("teacher_topk_ids" in item and "teacher_topk_probs" in item for item in batch):
            raise KeyError("either every batch item must contain teacher_topk_ids/probs or none of them may contain them")

        teacher_ids_batches: List[torch.Tensor] = []
        teacher_probs_batches: List[torch.Tensor] = []
        teacher_topk = None

        for item in batch:
            if not isinstance(item, dict):
                raise TypeError("batch items must be encoded dicts")
            required_keys = {"head_ids", "core_ids", "target_ids", "token_mask"}
            if not required_keys.issubset(item):
                raise KeyError("batch items must contain head_ids, core_ids, target_ids, token_mask")

            head_ids = list(item["head_ids"])
            core_ids = list(item["core_ids"])
            target_ids = list(item["target_ids"])
            token_mask = list(item["token_mask"])

            if len(head_ids) != 3:
                raise ValueError(f"head_ids length ({len(head_ids)}) must be 3")
            if len(core_ids) != len(target_ids) or len(core_ids) != len(token_mask):
                raise ValueError(
                    f"core_ids length ({len(core_ids)}) must equal target_ids length ({len(target_ids)}) "
                    f"and token_mask length ({len(token_mask)})"
                )

            head_batches.append(head_ids)
            core_batches.append(core_ids)
            target_batches.append(target_ids)
            token_mask_batches.append(token_mask)
            if has_sample_index:
                sample_indices.append(int(item["sample_index"]))

            if has_knowledge:
                ids = torch.as_tensor(item["teacher_topk_ids"], dtype=torch.int16)
                probs = torch.as_tensor(item["teacher_topk_probs"], dtype=torch.float16)
                if ids.dim() != 2 or probs.dim() != 2:
                    raise ValueError("teacher_topk_ids/probs must have shape (core_len, topk)")
                if ids.shape != probs.shape:
                    raise ValueError(f"teacher_topk_ids shape {tuple(ids.shape)} must match probs shape {tuple(probs.shape)}")
                if ids.size(0) != len(core_ids):
                    raise ValueError(
                        f"teacher knowledge length ({ids.size(0)}) must match core length ({len(core_ids)})"
                    )
                if teacher_topk is None:
                    teacher_topk = ids.size(1)
                elif teacher_topk != ids.size(1):
                    raise ValueError("all teacher knowledge rows in a batch must use the same topk")
                teacher_ids_batches.append(ids)
                teacher_probs_batches.append(probs)

        max_core_len = max(len(core_ids) for core_ids in core_batches)

        padded_core_batches: List[List[int]] = []
        padded_target_batches: List[List[int]] = []
        padded_token_mask_batches: List[List[int]] = []
        core_attention_mask_batches: List[List[int]] = []
        attention_mask_batches: List[List[int]] = []
        token_type_batches: List[List[int]] = []

        padded_teacher_ids: List[torch.Tensor] = []
        padded_teacher_probs: List[torch.Tensor] = []

        for row_idx, (head_ids, core_ids, target_ids, token_mask) in enumerate(
            zip(head_batches, core_batches, target_batches, token_mask_batches)
        ):
            pad_len = max_core_len - len(core_ids)
            padded_core_ids = core_ids + [self.pad_id] * pad_len
            padded_target_ids = target_ids + [self.pad_id] * pad_len
            padded_token_mask = token_mask + [0] * pad_len
            core_attention_mask = [1 if token_id != self.pad_id else 0 for token_id in padded_core_ids]

            sequence_ids = head_ids + padded_core_ids
            attention_mask = [1, 1, 1] + core_attention_mask
            token_type_ids = self._build_token_type_ids(sequence_ids)

            padded_core_batches.append(padded_core_ids)
            padded_target_batches.append(padded_target_ids)
            padded_token_mask_batches.append(padded_token_mask)
            core_attention_mask_batches.append(core_attention_mask)
            attention_mask_batches.append(attention_mask)
            token_type_batches.append(token_type_ids)

            if has_knowledge:
                ids = teacher_ids_batches[row_idx]
                probs = teacher_probs_batches[row_idx]
                if pad_len > 0:
                    ids_pad = torch.zeros((pad_len, teacher_topk), dtype=torch.int16)
                    probs_pad = torch.zeros((pad_len, teacher_topk), dtype=torch.float16)
                    ids = torch.cat([ids, ids_pad], dim=0)
                    probs = torch.cat([probs, probs_pad], dim=0)
                padded_teacher_ids.append(ids)
                padded_teacher_probs.append(probs)

        result: Dict[str, torch.Tensor] = {
            "head_ids": torch.tensor(head_batches, dtype=torch.long),
            "core_ids": torch.tensor(padded_core_batches, dtype=torch.long),
            "target_ids": torch.tensor(padded_target_batches, dtype=torch.long),
            "token_mask": torch.tensor(padded_token_mask_batches, dtype=torch.long),
            "core_attention_mask": torch.tensor(core_attention_mask_batches, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batches, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_batches, dtype=torch.long),
        }
        if has_sample_index:
            result["sample_index"] = torch.tensor(sample_indices, dtype=torch.long)
        if has_knowledge:
            result["teacher_topk_ids"] = torch.stack(padded_teacher_ids, dim=0)
            result["teacher_topk_probs"] = torch.stack(padded_teacher_probs, dim=0)
        return result

    def decode(self, token_ids):
        if not token_ids:
            return ""
        if isinstance(token_ids[0], list):
            return [self.decode(ids) for ids in token_ids]
        tokens = [self.id_to_token.get(int(token_id), "*") for token_id in token_ids]
        return "".join(tokens)
