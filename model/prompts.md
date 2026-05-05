# SarkazTokenizer 设计文档

## 核心概念

输入 `i` 和输出 `o` 除了特殊字符 [ ] | ; 外均一一对应。特殊字符在输入侧映射到分隔符，在输出侧映射到 0。
最终产出为三级编码：sample-level encode 返回三项（input_ids, target_ids, token_mask），batch-level collate 返回五项（input_ids, target_ids, token_mask, attention_mask, token_type_ids）。

## __init__

1. 读取词表到内存，词表默认路径 data/vocab.txt
2. 定义特殊 token id：
   - pad_id = 0
   - unk_id = 100
   - cls_id = 101
   - sep_id = 102
3. 定义魔数 token（根据 t 值选择）：
   - magic_id_0 = 6432 (当 t=0 时使用)
   - magic_id_1 = 5031 (当 t=1 时使用)
4. 初始化输入侧字符到 token id 的映射（使用 BERT 预定义的 unused 区间 1-99）：
   - a-z: 1-26
   - [: 27
   - ]: 28
   - |: 29
   - ;: 102 (sep_id，分隔符显化表示)
   - 注意：每个字符严格一对一映射，无贪心匹配（如 "hh"），保证输入输出长度一致

## encode(t: int, i: str, o: str) -> Dict[str, List[int]]

统一的样本级编码入口。

**参数**：
- t: 类型标记（0 或 1），用于选择 magic token
- i: 输入文本，只能包含 a-z [ ] | ; 和空格，将归一化（去空格+小写）
- o: 输出文本，用词表映射，特殊字符映射到 0

**处理流程**：
1. 验证 t 为 0 或 1，选择对应的 magic_id
2. 归一化输入文本 i（去空格、转小写），逐字符按 _input_char_to_id 映射生成 input_core（1-to-1 映射）
3. 编码输出文本 o，得到 target_ids（特殊字符 [ ] | ; 映射到 0，其他用词表查询；保证长度与 input_core 相同）
4. 生成 token_mask，特殊字符位置为 0，其余为 1
5. 构造 input_ids：[cls, magic, sep, ...input_core, sep]，长度自动比 target_ids 多 4

**返回**：
```python
{
    "input_ids": list[int],     # 含前缀 [cls, magic, sep] 和后缀 [sep]
    "target_ids": list[int],    # 与 input_core 同长
    "token_mask": list[int],    # 1 表示有效字符，0 表示特殊字符
}
```

## encode_target(text: str) -> List[int]

目标文本编码，使用词表映射。

**处理**：
- 归一化输入（去空格、转小写）
- 向后匹配 hh
- 特殊字符 [ ] | ; 映射到 0
- 其他字符用 token_to_id 词表查询，未找到则映射到 unk_id

## collate(batch: List[Dict]) -> Dict[str, torch.Tensor]

批处理，将已编码样本合并为张量。

**验证**：
- 每个样本必须包含 input_ids, target_ids, token_mask 三项
- batch 的最大 input 长度必须等于最大 target 长度 + 4

**处理**：
1. 找到 batch 中的最大 input 长度和最大 target 长度
2. 各样本分别 pad 到对应长度（input pad 到 max_input_len，target pad 到 max_target_len）
3. 生成 attention_mask：原始长度位置为 1，padding 位置为 0
4. 生成 token_type_ids：按 BERT 分段规则，遇到 sep 切换 segment id

**返回**：
```python
{
    "input_ids": torch.Tensor,        # shape (batch_size, max_input_len)
    "target_ids": torch.Tensor,       # shape (batch_size, max_target_len)
    "token_mask": torch.Tensor,       # shape (batch_size, max_target_len)
    "attention_mask": torch.Tensor,   # shape (batch_size, max_input_len)，masking 用于遮蔽 padding
    "token_type_ids": torch.Tensor,   # shape (batch_size, max_input_len)，分段标记
}
```

## decode(token_ids)

解码方法（保持兼容）。

1. 将输入 id 对应到词表
2. 如果输出没有匹配，解析为"*"
3. 支持批量解码（列表输入）