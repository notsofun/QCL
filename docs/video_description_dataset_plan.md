# Video-description 数据集重建方案

## 现状判断

这几组实验的主要信号不是“量子层还没调好”，而是当前 video-description 数据质量不足：

- `Data/llava-video-0-30-academic/manifest.csv` 只有 106 条样本，但 caption 是 LLaVA-Video 的长段落式说明，很多样本来自同一类 cooking/tutorial 场景，视频主题和文本 token 高度相似。
- `results/qcl_video_text_xclip_16f_chunktext_raw_vs_quantum/raw/video_text_results.csv` 的测试集只有 13 个 gallery，随机 Top-1 已经是 1/13；raw XCLIP 在测试 Top-1 为 0，Top-5 只有 0.0769，说明 frozen encoder 都无法从当前 manifest 中形成可靠的正负样本排序。
- 同目录 quantum 结果的 train mean rank 明显改善但 test Top-1 仍为 0，属于小数据 + 重复场景上适配器记忆/扰动训练集，无法泛化到测试样本的典型形态。

因此优先级应该是：先构建更干净、短文本、短视频、低重复的 Hugging Face 数据集，再讨论 adapter、pooling、quantum residual scale 等管线参数。

## 推荐数据源

默认使用 Hugging Face 上的 `VLM2Vec/MSR-VTT`：

- 任务对口：text-to-video retrieval / video-text matching。
- 规模适中：`test_1k` 子集可以先做 256～1000 条快速实验，避免 LLaVA-Video 长文本和下载 shard 的复杂度。
- 标注形态合适：MSR-VTT 的 caption 通常是一句话，很多是 3～12 个词，和“视频三四十秒、文本几个词”的目标一致。
- 多样性比当前本地 cooking/tutorial dump 更高：20 个类别，视频 id 和 caption 都可以直接做去重。

如果后续需要扩大规模，再切到 `train_7k` 或 `train_9k`；如果需要更严格的高质量英文描述，可以把 VATEX 作为第二阶段候选，但 MSR-VTT 更适合作为当前 debugging baseline。

## 落地流程

1. **抽样与基础过滤**
   - 随机扫描 Hugging Face split，避免只取前 N 条导致类别偏置。
   - 保留 5～40 秒视频，默认上限不超过 40 秒。
   - 保留 3～12 个英文词的 caption，自动截到第一句/第一分句。
   - 去掉完全重复 caption、重复 video id，以及明显不适合快速实验的成人/裸露关键词。

2. **物化成本地 manifest**
   - 下载/复制视频到 `Data/msrvtt-short-video-description/videos/`。
   - 写出 `Data/msrvtt-short-video-description/manifest.csv`，字段保持为现有 trainer 需要的 `video_path,text`。
   - 写出 `curation_metadata.json` 记录 dataset、config、split、seed、过滤阈值和 skip reason，确保实验可复现。

3. **训练建议**
   - 先用 raw adapter 做 frozen encoder baseline，不要马上训练 quantum adapter。
   - 第一个 sanity run 建议 `--max-samples 256`，确认 raw Top-1/Top-5 高于随机，并检查 positive logit 是否系统性高于 top negative logit。
   - raw baseline 稳定后再跑 `--adapter compare`，否则 quantum 层调参没有意义。

示例命令：

```bash
python scripts/curate_hf_video_description_dataset.py \
  --dataset VLM2Vec/MSR-VTT \
  --config test_1k \
  --split test \
  --max-samples 256 \
  --scan-rows 1000 \
  --min-duration 5 \
  --max-duration 40 \
  --min-words 3 \
  --max-words 12 \
  --output-dir Data/msrvtt-short-video-description \
  --strict

python model/qcl_train_refactored.py \
  --task video_text \
  --manifest Data/msrvtt-short-video-description/manifest.csv \
  --feature_backend video_encoder \
  --adapter raw \
  --frame_sampling all \
  --video_pooling chunks \
  --text_pooling chunks \
  --video_chunk_stride 8 \
  --match_pooling logmeanexp \
  --match_temperature 0.07 \
  --epoch 1 \
  --batch-size 256 \
  --cache_features true \
  --result_path results/msrvtt_short_raw_sanity \
  --model_path results/msrvtt_short_raw_sanity_models \
  --device auto
```

## Sanity check 通过标准

脚本会在下载完成后自动生成 `sanity_report.json`。在正式训练前至少检查：

- `decode_failures` 为空：所有视频都能被 OpenCV 解码。
- `duplicate_text_groups` 为空：没有完全重复 caption。
- `duplicate_file_hash_groups` 为空：没有字节级重复视频。
- `duplicate_frame_hash_groups` 为空或很少：没有明显近重复视频。
- `duration_seconds.max <= 40` 且 `duration_seconds.min >= 5`。
- `word_counts.max <= 12` 且 `word_counts.min >= 3`。
- 样本数至少 128，推荐 256 起步；低于 32 条只适合 smoke test，不适合判断模型优劣。

训练后的最低 sanity 标准：

- raw baseline 的 Top-k 应该高于随机 Top-k，尤其 Top-5 应明显超过 `5 / num_gallery`。
- `test_margin_mean` 不应该长期为大负数；如果 positive logit 平均仍低于 top negative logit，优先回看数据，而不是调 quantum adapter。
- train 指标提升但 val/test 不动时，优先检查重复主题、重复画面、split 过小，而不是增加 epoch。
