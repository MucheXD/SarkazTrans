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

        # 保证所有行宽度一致，较短的行用 False 填充
        width = max(len(r) for r in rows)
        for r in rows:
            if len(r) < width:
                r.extend([False] * (width - len(r)))

        self.chars = chars
        self.char_to_index = {c: i for i, c in enumerate(self.chars)}
        self.output = torch.tensor(rows, dtype=torch.bool)

        # 输入侧 core token 使用 1-26 表示 a-z；其余 id 视为无有效映射。
        # 这里映射 100 是为了方便之后添加其它 token, 实际上多余的部分最后都是 -1
        self._input_id_to_row_index = torch.full((100,), -1, dtype=torch.long)
        for idx in range(1, min(len(self.chars), 26) + 1):
            self._input_id_to_row_index[idx] = idx - 1

    def index_of(self, ch: str) -> int:
        """返回字符对应的行索引，找不到抛出 KeyError。"""
        return self.char_to_index[ch]

    def map_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将 input_ids 的 core 区间映射成 (batch, len[3:-1], dict_size) 的布尔张量。"""
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, len)")

        core_input_ids = input_ids[:, 3:-1].long()
        batch_size, seq_len = core_input_ids.shape
        dict_size = self.output.size(1)

        if seq_len == 0:
            return torch.zeros((batch_size, 0, dict_size), dtype=torch.bool, device=core_input_ids.device)

        device = core_input_ids.device
        lookup = self._input_id_to_row_index.to(device)
        output = self.output.to(device)
        
        safe_input_ids = core_input_ids.clamp(min=0, max=lookup.size(0) - 1)
        row_indices = lookup[safe_input_ids]

        mapped = output[row_indices.clamp(min=0)]
        invalid_mask = row_indices < 0
        if invalid_mask.any():
            mapped = mapped.masked_fill(invalid_mask.unsqueeze(-1), False)

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
                    
                if len(encoded_sample["input_ids"]) > max_length:
                    # Keep CLS, magic, SEP, and as much core as possible, then the trailing SEP.
                    core_max = max_length - 4
                    if core_max <= 0:
                        print(f"⚠️  Skipping line {line_no}: max_length={max_length} too small")
                        continue

                    core_len = min(len(encoded_sample["token_mask"]), core_max)

                    # 严格验证截断长度一致性
                    target_core = encoded_sample["target_ids"][:core_len]
                    mask_core = encoded_sample["token_mask"][:core_len]
                    assert len(target_core) == core_len and len(mask_core) == core_len, \
                        f"Line {line_no}: Truncation length mismatch after slicing (target={len(target_core)}, mask={len(mask_core)}, expected={core_len})"
                    
                    encoded_sample = {
                        "input_ids": encoded_sample["input_ids"][:3 + core_len] + [tokenizer.sep_id],
                        "target_ids": target_core,
                        "token_mask": mask_core,
                    }
                    
                    # 验证截断后形状
                    assert len(encoded_sample["input_ids"]) == 4 + core_len, \
                        f"Line {line_no}: After truncation, input_ids should have length {4 + core_len}, got {len(encoded_sample['input_ids'])}"
                    assert len(encoded_sample["target_ids"]) == core_len and len(encoded_sample["token_mask"]) == core_len, \
                        f"Line {line_no}: After truncation, target_ids and token_mask should have length {core_len}"

                self.data.append(encoded_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]