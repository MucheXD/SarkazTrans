# collect.py

你需要编写 Python 脚本，从我指定的网站获取数据并储存在本地。

1. GET https://warfarin.wiki/cn/missions.data 获取原始响应数据并储存
2. 调用 wikidat_deserialize 反序列化
3. 根据反序列化结果，遍历 data 下的 object，其中有 id 字段的，进行下述请求
4. GET https://warfarin.wiki/cn/missions/{{id}}.data 获取原始响应数据并储存
5. 调用反序列化

注意：

- 请求及反序列化样例可以在 data/example 下找到
- 原始数据请储存在 data/raw 下，反序列化的数据请储存在 data/json 下
- 请求网站的频率不宜过高，每次请求间隔至少 200ms
- 第一步请求时如果有 set-cookie，那么存下来在后续请求带上
- 如果获取某个文件数据失败，优先重试，重试失败后一定要有日志，以确保完整性
- 脚本最终暴露 collect_all_data 函数执行所述步骤

- 默认情况下，如果 raw 中已经存在 同名文件，则不执行这一份 missions 的网络获取，如果 json 中不存在，直接从 raw 反序列化
- 如果传入 force_recollect=True 那么忽略上述规则，总是重新获取并反序列化
- 总是请求 missions.data，否则你不知道哪些是需要下载的

# convert.py

你需要编写 Python 脚本，将存储的 JSON 数据转换为训练数据集。

1. 你需要遍历读取 data/json 下的所有 JSON 文件
2. 定位到 data/dialog （Array），遍历其中的 Object
3. 按规则提取 type、dialogText 两个字段，按照规则序列化为训练数据

序列化规则：
以下是你的序列化目标：
```
{"t": 0,"i": "by[jg]rhq","o": "负责训练的教官"}, // 一条目一行
...
```
其中 t 是类别，i 是输入，o 是目标输出（标签）

t 根据如下规则转换：
- type=dialog -> 0
- type=summary -> 0
- type=option -> 1

i 需要你将 dialogText 投入 encode encode_text 函数得到结果
o 则是 dialogText 的文本，但需要去除标记符号，例如（... 表示任意文本，可以参考 encode.py 的做法）：
<@...>
</>
<...HTML标签...>

注意：
- 注意设置断言，务必保证训练数据质量
- 如果遇到错误，直接输出日志并终止，让我介入处理特殊情况

# decode.py

你需要编写 Python 脚本，对 encode 后的 26 个值反向查找可能的中文字符。

1. 输入某个 encode 后的 token（不包括 [ ] | 三个特殊字符）
2. 读取字表 vocab.txt
3. 对输入 token，遍历字表，如果字表该字的 encode 结果等于 token，那么当前位置为 1
4. 你的输出应该是 26*(vocab_size+1) 的矩阵，输出到文本 map.txt，类似:
a 1 0 1 0 0 ... 0 1
b 0 0 0 1 0 ... 1 0
...