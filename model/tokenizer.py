from pathlib import Path
from typing import Dict, List

import torch

class SarkazTokenizer:
    def __init__(self, vocab_file=None):

        # 定位词表
        base_dir = Path(__file__).resolve().parent
        default_vocab = base_dir / "data" / "vocab.txt"
        self.vocab_file = Path(vocab_file) if vocab_file else default_vocab

        # 加载词表 获得 id<->token 映射
        self.id_to_token = {}
        self.token_to_id = {}
        with self.vocab_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        # 预定义特殊字符的 id
        self.pad_id = 0
        self.unk_id = 100 # 此处 unk 仅用于 target 编码，输入侧的未知字符直接映射到 0（pad_id）以保证不产生有效 token
        self.cls_id = 101
        self.sep_id = 102
        # 此处的两个魔数用于区分输入类别 将类别信息提供给模型
        # 为了最大化 bert 的预训练，魔数选择了语义最匹配的汉字
        self.magic_id_0 = 6432 # 对应 "说" 表示对话
        self.magic_id_1 = 5031 # 对应 "答" 表示选项

        # a-z: 1-26, [: 27, ]: 28, |: 29
        # 每个字符严格一对一映射
        self._input_char_to_id = {
            "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
            "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
            "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
            "[": 27, "]": 28, "|": 29,
        }
        self._special_target_tokens = {"[", "]", "|", ";"}

    def _normalize(self, text):
        return "".join(text.split()).lower()

    def _encode_core(self, text):
        '''
        Encodes the input text into a list of token IDs and a corresponding mask.
        '''
        
        normalized = self._normalize(text)

        input_core = []
        token_mask = []

        for ch in normalized:
            input_core.append(self._input_char_to_id.get(ch, 0))
            if ch in self._special_target_tokens:
                token_mask.append(0)
            else:
                token_mask.append(1)

        return input_core, token_mask

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

        core_ids, token_mask = self._encode_core(i)

        head_ids = [self.cls_id, magic_id, self.sep_id]
        target_ids = self.encode_target(o)

        # Verify contract once at sample creation time: valid output tokens must match usable input positions.
        valid_token_count = sum(token_mask)
        if valid_token_count != len(target_ids):
            raise ValueError(
                f"Valid token count ({valid_token_count}) must equal target_ids length ({len(target_ids)}). "
                f"Input: '{i}' -> {len(core_ids)} core tokens, {valid_token_count} valid. "
                f"Output: '{o}' -> {len(target_ids)} tokens."
            )
        
        return {
            "head_ids": head_ids,
            "core_ids": core_ids,
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
        assert batch, "batch must not be empty"

        head_batches = []
        core_batches = []
        target_batches = []
        token_mask_batches = []

        # 将传入的键值对重装成 batch 块（未对齐）
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
                    f"core_ids length ({len(core_ids)}) must be target_ids length ({len(target_ids)}) and token_mask length ({len(token_mask)})"
                )

            head_batches.append(head_ids)
            core_batches.append(core_ids)
            target_batches.append(target_ids)
            token_mask_batches.append(token_mask)

        max_core_len = max(len(core_ids) for core_ids in core_batches)
        max_target_len = max(len(target_ids) for target_ids in target_batches)
        if max_core_len != max_target_len:
            raise ValueError(f"core length ({max_core_len}) must match target length ({max_target_len})")

        padded_core_batches = []
        padded_target_batches = []
        padded_token_mask_batches = []
        attention_mask_batches = []
        token_type_batches = []

        for head_ids, core_ids, target_ids, token_mask in zip(head_batches, core_batches, target_batches, token_mask_batches):
            pad_len = max_core_len - len(core_ids)
            padded_core_ids = core_ids + [self.pad_id] * pad_len
            padded_target_ids = target_ids + [0] * pad_len
            padded_token_mask = token_mask + [0] * pad_len

            sequence_ids = head_ids + padded_core_ids
            attention_mask = [1 if token_id != self.pad_id else 0 for token_id in sequence_ids]
            token_type_ids = self._build_token_type_ids(sequence_ids)

            padded_core_batches.append(padded_core_ids)
            padded_target_batches.append(padded_target_ids)
            padded_token_mask_batches.append(padded_token_mask)
            attention_mask_batches.append(attention_mask)
            token_type_batches.append(token_type_ids)

        return {
            "head_ids": torch.tensor(head_batches, dtype=torch.long),
            "core_ids": torch.tensor(padded_core_batches, dtype=torch.long),
            "target_ids": torch.tensor(padded_target_batches, dtype=torch.long),
            "token_mask": torch.tensor(padded_token_mask_batches, dtype=torch.long),
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
