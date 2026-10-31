# 英中 Transformer：训练完成原始存档

本仓库保存 2026 年 10 月 31 日在 AutoDL 完成 20 轮训练时的原始版本。代码、分词器、配置、日志与当时导出包一致，保留当时的实现，不加入后续的优化或实验报告评估代码。

## 原始文件

- `model.ipynb`：当时的 Notebook。
- `train_autodl.py`：当时实际使用的训练脚本。
- `tokenizer_opus100_char_v1_en.json`、`tokenizer_opus100_char_v1_zh.json`：当时的英中分词器。
- `trained_config.json`、`training_status.json`：训练配置和完成记录。
- `training.log`、`runs/`：原始日志及 TensorBoard 记录。
- `validation_results.json`：训练完成时保存的原始示例结果。
- `trained_export_manifest.json`：当时的文件校验清单。

## 完整模型存档

在 [训练完成原始版 Release](https://github.com/chabingcha/english-chinese-transformer-original/releases/tag/training-complete-20261031) 下载：

1. `autodl_en_zh_trained_20261031.tar`：原始导出包，含最终权重 `weights_opus100_en_zh_char_v1/tmodel_19.pt`（模型参数及优化器等训练状态）。
2. `original_data_20261031.tar.gz`：原样数据补充包。
3. `SHA256SUMS.txt`：附件校验值。

将两个压缩包解压到同一个目录，即可合并得到带原始数据的 `autodl_en_zh_trained_20261031` 文件夹。权重不直接存入 Git 历史，完整原始包保存在 Release 附件中。

原始导出包 SHA256：

```
192fd99cf304e532bdea969a1d977f0958dc3cc603f6a5d368ce3e9f9e4c7e03
```

数据补充包里原有的 `tokenizer.model`、`tokenizer.vocab` 是旧的数据准备文件；这次模型实际使用上面列出的两份 `tokenizer_opus100_char_v1_*.json`。

## 当时的训练设置

- 英语到中文，OPUS-100 本地样本；20,000 对训练数据经长度过滤后使用 19,991 对。
- Encoder 6 层、Decoder 6 层，`d_model=512`，8 个注意力头，`d_ff=2048`。
- Batch size 8，Adam，学习率 0.0001，序列长度 350，训练 20 轮。
- 最终检查点为 `tmodel_19.pt`；本存档仅保留最终检查点，没有打包前 19 轮中间权重。
- 当时环境：Python 3.12，PyTorch 2.8.0+cu128，AutoDL 实例报告 NVIDIA GeForce RTX 4090。
- 原始程序依赖 PyTorch、tokenizers、datasets、tensorboard、tqdm 等，Notebook 需要 Jupyter 环境。
- 本仓库作为原样存档保存；运行训练脚本或全部执行 Notebook 会触发原程序中的训练流程。

## 参考来源

- [Umar Jamil：Coding a Transformer from scratch on PyTorch](https://www.youtube.com/watch?v=ISNdQcPhsts)
- [《动手学深度学习》Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)
- [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)，数据版本 `805090dc28bf78897da9641cdf08b61287580df9`。数据及参考实现的权利归原作者及相应来源所有，本存档未额外赋予许可。
