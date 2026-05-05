from pathlib import Path
from typing import Dict, List

import torch

class SarkazTokenizer:
    def __init__(self, vocab_file=None):
        base_dir = Path(__file__).resolve().parent
        default_vocab = base_dir / "data" / "vocab.txt"
        self.vocab_file = Path(vocab_file) if vocab_file else default_vocab

        self.id_to_token = {}
        self.token_to_id = {}
        with self.vocab_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                self.id_to_token[idx] = token
                self.token_to_id[token] = idx

        self.pad_id = 0
        self.unk_id = 100
        self.cls_id = 101
        self.sep_id = 102
        # 此处的两个魔数用于区分输入类别 将类别信息提供给模型
        # 为了最大化 bert 的预训练，魔数选择了语义最匹配的汉字
        self.magic_id_0 = 6432 # 对应 "说" 表示对话
        self.magic_id_1 = 5031 # 对应 "答" 表示选项

        # 输入侧使用 BERT 预定义的 unused 区间（1-99）进行映射
        # a-z: 1-26, [: 27, ]: 28, |: 29, ;: sep_id（分隔符显化表示）
        # 每个字符严格一对一映射
        self._input_char_to_id = {
            "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
            "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
            "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
            "[": 27, "]": 28, "|": 29, ";": self.sep_id,
        }
        self._special_target_tokens = {"[", "]", "|", ";"}

    def _normalize(self, text):
        return "".join(text.split()).lower()

    def _encode_core(self, text):
        normalized = self._normalize(text)

        input_core = []
        target_core = []
        token_mask = []

        for ch in normalized:
            input_core.append(self._input_char_to_id.get(ch, self.unk_id))

            if ch in self._special_target_tokens:
                target_core.append(0)
                token_mask.append(0)
            else:
                target_core.append(self.token_to_id.get(ch, self.unk_id))
                token_mask.append(1)

        return input_core, target_core, token_mask

    def _build_token_type_ids(self, ids):
        type_ids = []
        segment_id = 0
        for token_id in ids:
            type_ids.append(segment_id)
            if token_id == self.sep_id:
                segment_id = 1
        return type_ids

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

        input_core, _, token_mask = self._encode_core(i)

        input_ids = [self.cls_id, magic_id, self.sep_id, *input_core, self.sep_id]
        raw_target_ids = self.encode_target(o)

        # Verify contract once at sample creation time: valid output tokens must match usable input positions.
        valid_token_count = sum(token_mask)
        if valid_token_count != len(raw_target_ids):
            raise ValueError(
                f"Valid token count ({valid_token_count}) must equal target_ids length ({len(raw_target_ids)}). "
                f"Input: '{i}' -> {len(input_core)} core tokens, {valid_token_count} valid. "
                f"Output: '{o}' -> {len(raw_target_ids)} tokens."
            )

        target_ids = []
        target_index = 0
        for mask in token_mask:
            if mask == 1:
                target_ids.append(raw_target_ids[target_index])
                target_index += 1
            else:
                target_ids.append(0)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "token_mask": token_mask,
        }

    def encode_target(self, text: str) -> List[int]:
        """Encode target text using the vocabulary mapping.
        Each character maps to exactly one token (1-to-1 mapping, no greedy matching).
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")

        normalized = self._normalize(text)
        ids = []
        for ch in normalized:
            ids.append(self.token_to_id.get(ch, self.unk_id))
        return ids

    def collate(self, batch) -> Dict[str, torch.Tensor]:
        
        assert isinstance(batch, list), "batch must be a list of encoded samples"
        assert len(batch) > 0, "batch must not be empty"

        encoded_samples = []
        for item in batch:
            if not isinstance(item, dict):
                raise TypeError("batch items must be encoded dicts")
            if not {"input_ids", "target_ids", "token_mask"}.issubset(item):
                raise KeyError("batch items must contain input_ids, target_ids, token_mask")
            
            # 验证各部分形状一致
            input_len = len(item["input_ids"])
            target_len = len(item["target_ids"])
            mask_len = len(item["token_mask"])
            assert target_len == mask_len, f"target_ids ({target_len}) and token_mask ({mask_len}) length mismatch"
            assert input_len == target_len + 4, \
                f"input_ids length ({input_len}) must be target_ids length ({target_len}) + 4 (CLS, magic, SEP, SEP), got difference {input_len - target_len}"
            
            encoded_samples.append({
                "input_ids": item["input_ids"],
                "target_ids": item["target_ids"],
                "token_mask": item["token_mask"],
            })

        input_lengths = [len(item["input_ids"]) for item in encoded_samples]
        target_lengths = [len(item["target_ids"]) for item in encoded_samples]
        max_input_len = max(input_lengths)
        max_target_len = max(target_lengths)

        input_batches = []
        target_batches = []
        token_mask_batches = []
        attention_mask_batches = []
        token_type_batches = []

        for item in encoded_samples:
            input_ids = item["input_ids"]
            target_ids = item["target_ids"]
            token_mask = item["token_mask"]

            assert len(input_ids)-4 == len(target_ids) == len(token_mask), "input_ids length must be target_ids length + 4 (for CLS, magic, SEP, SEP)"

            input_pad_len = max_input_len - len(input_ids)
            target_pad_len = max_target_len - len(target_ids)

            assert target_pad_len == input_pad_len, "Input and target must be padded to the same length"

            padded_input_ids = input_ids + [self.pad_id] * input_pad_len
            padded_target_ids = target_ids + [0] * target_pad_len
            padded_token_mask = token_mask + [0] * target_pad_len
            attention_mask = [1] * len(input_ids) + [0] * input_pad_len

            token_type_ids = []
            segment_id = 0
            for token_id in padded_input_ids:
                token_type_ids.append(segment_id)
                if token_id == self.sep_id:
                    segment_id = 1

            input_batches.append(padded_input_ids)
            target_batches.append(padded_target_ids)
            token_mask_batches.append(padded_token_mask)
            attention_mask_batches.append(attention_mask)
            token_type_batches.append(token_type_ids)

        return {
            "input_ids": torch.tensor(input_batches, dtype=torch.long),
            "target_ids": torch.tensor(target_batches, dtype=torch.long),
            "token_mask": torch.tensor(token_mask_batches, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batches, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_batches, dtype=torch.long),
        }

    def decode(self, token_ids):
        if not token_ids:
            return ""

        if isinstance(token_ids[0], list):
            return [self.decode(ids) for ids in token_ids]

        tokens = [self.id_to_token.get(token_id, "*") for token_id in token_ids]
        return "".join(tokens)
