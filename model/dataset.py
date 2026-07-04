import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizer import SarkazTokenizer


class SarkazCharmap:
    """Character-condition mask loaded from data/map.txt.

    The returned mask has shape (batch, seq_len, 8100). For input core ids 1..26
    it contains the allowed student-vocabulary ids for that source character.
    For padding, delimiters, and unknown ids it returns all False.
    """

    def __init__(self, map_path: str = "data/map.txt"):
        map_file = Path(map_path)
        if not map_file.is_absolute():
            map_file = Path(__file__).resolve().parent / map_file

        chars = []
        rows = []

        with map_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"无法解析 {map_file} 第 {line_no} 行：格式错误")

                ch = parts[0]
                vals = parts[1:]

                try:
                    bools = [bool(int(x)) for x in vals]
                except Exception as exc:
                    raise ValueError(f"无法解析 {map_file} 第 {line_no} 行的像素数据") from exc

                chars.append(ch)
                rows.append(bools)

        if not rows:
            raise ValueError(f"地图文件 {map_file} 为空或未找到有效行")

        width = max(len(r) for r in rows)
        for r in rows:
            if len(r) != width:
                raise ValueError(f"地图文件 {map_file} 存在不等宽行")

        if len(chars) != 26:
            raise ValueError(f"地图文件 {map_file} 应包含 26 行 a-z 映射，实际 {len(chars)} 行")

        self.chars = chars
        self.char_to_index = {c: i for i, c in enumerate(self.chars)}
        self.output = torch.tensor(rows, dtype=torch.bool)

        self._input_id_to_row_index = torch.full((27,), -1, dtype=torch.long)
        for idx in range(1, 26 + 1):
            self._input_id_to_row_index[idx] = idx - 1

    def index_of(self, ch: str) -> int:
        return self.char_to_index[ch]

    def map_core_ids(self, core_ids: torch.Tensor) -> torch.Tensor:
        """Map compact core ids to allowed-output masks.

        Args:
            core_ids: Long tensor of shape (batch, seq_len).

        Returns:
            Bool tensor of shape (batch, seq_len, 8100).
        """
        if not torch.is_tensor(core_ids):
            core_ids = torch.tensor(core_ids, dtype=torch.long)
        if core_ids.dim() != 2:
            raise ValueError("core_ids 应该是 (batch_size, seq_len) 的二维张量")

        device = core_ids.device
        lookup = self._input_id_to_row_index.to(device)
        output = self.output.to(device)

        core_ids_long = core_ids.long()
        in_lookup_range = (core_ids_long >= 0) & (core_ids_long < lookup.size(0))
        safe_core_ids = core_ids_long.clamp(min=0, max=lookup.size(0) - 1)
        row_indices = lookup[safe_core_ids]
        valid_rows = in_lookup_range & (row_indices >= 0)

        safe_row_indices = row_indices.clamp(min=0)
        mapped = output[safe_row_indices]
        mapped = mapped.clone()
        mapped[~valid_rows] = False
        return mapped


class SarkazDataset(Dataset):
    def __init__(self, raw_data: str, tokenizer: SarkazTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        raw_path = Path(raw_data)
        if not raw_path.is_absolute():
            raw_path = Path(__file__).resolve().parent / raw_path
        self.raw_path = raw_path

        self.data = []

        with raw_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"无法解析 {raw_path} 第 {line_no} 行") from exc

                if not isinstance(item, dict):
                    raise TypeError(f"{raw_path} 第 {line_no} 行必须是对象")

                try:
                    t = item["t"]
                    input_text = item["i"]
                    output_text = item["o"]
                except KeyError as exc:
                    raise KeyError(f"{raw_path} 第 {line_no} 行缺少字段: {exc.args[0]}") from exc

                if not isinstance(t, int):
                    raise TypeError(f"{raw_path} 第 {line_no} 行的 t 必须是整数")
                if not isinstance(input_text, str):
                    raise TypeError(f"{raw_path} 第 {line_no} 行的 i 必须是字符串")
                if not isinstance(output_text, str):
                    raise TypeError(f"{raw_path} 第 {line_no} 行的 o 必须是字符串")

                try:
                    encoded_sample = tokenizer.encode(t, input_text, output_text)
                except ValueError as exc:
                    print(f"Skipping line {line_no}: {exc}")
                    continue

                head_len = len(encoded_sample["head_ids"])
                if head_len > max_length:
                    print(f"Skipping line {line_no}: max_length={max_length} too small for head_ids length={head_len}")
                    continue

                core_len = len(encoded_sample["core_ids"])
                if head_len + core_len > max_length:
                    core_max = max_length - head_len
                    if core_max <= 0:
                        print(f"Skipping line {line_no}: max_length={max_length} too small")
                        continue

                    core_len = min(
                        len(encoded_sample["core_ids"]),
                        len(encoded_sample["target_ids"]),
                        len(encoded_sample["token_mask"]),
                        core_max,
                    )
                    encoded_sample = {
                        "head_ids": encoded_sample["head_ids"],
                        "core_ids": encoded_sample["core_ids"][:core_len],
                        "target_ids": encoded_sample["target_ids"][:core_len],
                        "token_mask": encoded_sample["token_mask"][:core_len],
                    }

                if not (
                    len(encoded_sample["core_ids"])
                    == len(encoded_sample["target_ids"])
                    == len(encoded_sample["token_mask"])
                ):
                    raise AssertionError(f"Line {line_no}: core_ids/target_ids/token_mask length mismatch")

                encoded_sample["sample_index"] = len(self.data)
                self.data.append(encoded_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
