# SarkazTrans 萨卡兹语译者

## 项目简介

“萨卡兹”（Sarkaz）是游戏《明日方舟》（Arknights）、《明日方舟：终末地》（Arknights: Endfield）及衍生 IP 中的虚构种族。“萨卡兹语”在本项目中指上述游戏及其衍生 IP 中使用的一种符号文字，由 26 个字母组成。“萨卡兹语”不是鹰角网络（HYPERGRYPH）官方确定的符号体系，在其不同的作品中具有不同的编码方式。

本项目主要针对从中文转换的编码方式，该编码方式被广泛用于《明日方舟：终末地》的游戏界面、动画与效果图中，其正向编码算法最初由 [@比力币](https://space.bilibili.com/488486599) 发现（可参考外部视频 [我们破解了终末地的萨卡兹语！](https://www.bilibili.com/video/BV1PtdyBMEED)，在此感谢其对社区的贡献）。由于这种映射关系将每个中文字符对应到 26 种符号之一，其信息压缩极其严重，没有办法通过某种逆向算法进行确定性转换。因此，本项目试图利用深度学习的方法，通过深度神经网络给出原文字符位置的概率分布，进而为后续的推测提供空间。

## 项目实现

本项目使用 Bert 作为基座模型，通过增加映射层、缩减词空间等方式特化模型，结合蒸馏法、分层微调等技巧训练模型，目标是让正确的词元在预测结果中尽可能靠前。

在最新的 v4 版本中，模型经过初级微调能够在通用语料验证集上取得 85.60% 的 Top5 逐词元准确率。由于领域语料收集和算力的原因，暂时还没有进行针对游戏领域语料的微调训练。不过，由于领域语料涉及面相对较窄，在收集到足够多有效语料并对专有名词强化训练的情况下，预计能获得高于通用语料的准确率。

## 实机训练参考

GPU: GeForce RTX 5060 (8+8GB)

特别感谢我的某位小伙伴无偿提供的算力资源！

| 模型  | 至最佳时间 | 最佳 Epoch | 最佳 Top1 | 最佳 Top5 | 结束原因 | 提升点       |
| --- | ----- | -------- | ------- | ------- | ---- | --------- |
| v1  | 1.7h  | 8        | 0.456   | 0.614   | 过拟合  | -         |
| v2  | 2.4h  | 12       | 0.504   | 0.679   | 过拟合  | 增加适应层     |
| v3  | 13.9h | 43       | 0.526   | 0.687   | 趋平早停 | 修正泄露、差速训练 |
| v4  | 6.6h  | 24       | 0.650   | 0.856   | 趋平早停 | 引入蒸馏法     |

如果你想要自行训练，你需要补齐未提供的外部依赖。必要的文件及其目录结构可参考如下树状图：
```
.
├── model/
│   ├── bert-base-chinese/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── data/
│   │   ├── map.txt
│   │   ├── pretrain.jsonl
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── train.jsonl
│   ├── dataset.py
│   ├── knowledge-cache.py
│   ├── modelSaver.py
│   ├── sarkazBert.py
│   ├── summaryLogger.py
│   ├── teacher-model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── trainer.py
│   └── vocab.txt
└── teacher/
    └── bert-base-chinese/
        ├── config.json
        ├── model.safetensors
        ├── README.md
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── vocab.txt
```

## 项目结构

- `collect` 用于收集语料的脚本，领域语料的数据源为 [华法琳Wiki](warfarin.wiki)，出于版权考虑，本仓库不提供处理后数据
- `model` 训练与验证脚本，包括 Bert 相关性能测试的脚本等

本项目需要引入 bert-base-chinese 模型，你可以 [到 Hugging Face 上下载](https://huggingface.co/google-bert/bert-base-chinese)。