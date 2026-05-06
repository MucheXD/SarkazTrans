import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizer import SarkazTokenizer


class SarkazCharmap():
    """读取 data/map.txt 并构建字符到布尔图的索引。

    实例属性：
    - chars: 字符列表，顺序与文件一致
    - char_to_index: 字符 -> 行索引 的字典
    - output: `torch.BoolTensor`，形状 (num_chars, num_pixels)，可以通过 `output[i, :]` 取得第 i 个字符的布尔图
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

        # 保证所有行宽度一致
        width = max(len(r) for r in rows)
        for r in rows:
            assert len(r) == width

        # 验证合法 char 对应数量
        assert len(chars) == 26

        self.chars = chars
        self.char_to_index = {c: i for i, c in enumerate(self.chars)}
        self.output = torch.tensor(rows, dtype=torch.bool)

        # 输入侧 core token 使用 1-26 表示 a-z；其余 id 视为无有效映射。
        self._input_id_to_row_index = torch.full((27,), -1, dtype=torch.long)
        for idx in range(1, 26 + 1):
            self._input_id_to_row_index[idx] = idx - 1

    def index_of(self, ch: str) -> int:
        """返回字符对应的行索引，找不到抛出 KeyError。"""
        return self.char_to_index[ch]

    def map_core_ids(self, core_ids: torch.Tensor) -> torch.Tensor:
        """将 core_ids 映射成 (batch, seq_len, dict_size) 的布尔张量。"""
        if not torch.is_tensor(core_ids):
            core_ids = torch.tensor(core_ids, dtype=torch.long)

        assert core_ids.dim() == 2, "core_ids 应该是 (batch_size, seq_len) 的二维张量"

        # core_ids = core_ids.long()
        # batch_size, seq_len = core_ids.shape
        # dict_size = self.output.size(1)
        # assert seq_len != 0

        device = core_ids.device
        lookup = self._input_id_to_row_index.to(device)
        output = self.output.to(device)
        
        # safe_core_ids = core_ids.clamp(min=0, max=lookup.size(0) - 1)
        # 从 id 到 map idx
        row_indices = lookup[core_ids]
        # 从 map idx 布尔图
        mapped = output[row_indices]
        invalid_mask = row_indices < 0
        mapped[invalid_mask] = False
        return mapped

class SarkazDataset(Dataset):
    def __init__(self, raw_data: str,
                  tokenizer: SarkazTokenizer,
                  max_length = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        raw_path = Path(raw_data)
        if not raw_path.is_absolute():
            raw_path = Path(__file__).resolve().parent / raw_path

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
                except ValueError as e:
                    # Skip samples that don't match the contract
                    print(f"⚠️  Skipping line {line_no}: {e}")
                    continue
                    
                head_len = len(encoded_sample["head_ids"])
                if head_len > max_length:
                    print(f"⚠️  Skipping line {line_no}: max_length={max_length} too small for head_ids length={head_len}")
                    continue

                if head_len + len(encoded_sample["core_ids"]) > max_length:
                    # Keep head_ids fixed, and truncate the core part to fit max_length.
                    core_max = max_length - head_len
                    if core_max <= 0:
                        print(f"⚠️  Skipping line {line_no}: max_length={max_length} too small")
                        continue

                    core_len = min(len(encoded_sample["core_ids"]), len(encoded_sample["target_ids"]), len(encoded_sample["token_mask"]), core_max)

                    # 严格验证截断长度一致性
                    core_ids = encoded_sample["core_ids"][:core_len]
                    target_core = encoded_sample["target_ids"][:core_len]
                    mask_core = encoded_sample["token_mask"][:core_len]
                    assert len(core_ids) == core_len and len(target_core) == core_len and len(mask_core) == core_len, \
                        f"Line {line_no}: Truncation length mismatch after slicing (core={len(core_ids)}, target={len(target_core)}, mask={len(mask_core)}, expected={core_len})"
                    
                    encoded_sample = {
                        "head_ids": encoded_sample["head_ids"],
                        "core_ids": core_ids,
                        "target_ids": target_core,
                        "token_mask": mask_core,
                    }
                    
                    # 验证截断后形状
                    assert len(encoded_sample["head_ids"]) == head_len and len(encoded_sample["core_ids"]) == core_len, \
                        f"Line {line_no}: After truncation, head_ids should have length {head_len} and core_ids should have length {core_len}"
                    assert len(encoded_sample["target_ids"]) == core_len and len(encoded_sample["token_mask"]) == core_len, \
                        f"Line {line_no}: After truncation, target_ids and token_mask should have length {core_len}"

                self.data.append(encoded_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]