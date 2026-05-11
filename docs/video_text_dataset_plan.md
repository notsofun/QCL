# Video-description 数据集重建方案

## 结论：先换数据，再调管线

当前几组实验目录（`results/local_raw_xclip16_textchunks_dump*`、`results/qcl_video_text_xclip_16f_chunktext_raw_vs_quantum*`）反映出的核心风险不是量子/非量子 adapter，而是数据层：样本数小、视频内容重复、caption 过长且同质，导致 retrieval 指标接近随机或在很小验证集上虚高。此时继续调 `num_frames`、pooling 或 quantum residual，很容易只是在拟合噪声。

新的数据集目标应该是：

- **视频短**：优先 4-40 秒；若需要严格“三四十秒以内”，用 `--max-duration 40`。
- **文本短**：2-8 个英文词，避免 LLaVA 风格长段落描述。
- **语义分散**：体育、动物、烹饪、音乐、游戏、交通、人物动作都要覆盖，不能全是同一类 cooking clips。
- **去重先于训练**：exact hash、感知 hash、caption 去重必须在写 manifest 前完成。

## 推荐 Hugging Face 数据源

### 1. 首选：`VLM2Vec/MSR-VTT`

适合当前 video-text retrieval 管线的原因：

- 数据集页面标注为 Text-to-Video/Text Retrieval/Video Classification，并包含 `caption`、`url`、`start time`、`end time`、`category` 等列。
- MSR-VTT 本身就是经典 video-description/retrieval 数据集，caption 通常是短句，比现有 LLaVA 长段落更适合对比学习。
- Hugging Face viewer 中的样例片段多为约 10-25 秒，符合“不要太长”的要求。

注意：这个版本通常给 YouTube URL 和时间戳，真正落地视频需要 `yt-dlp`。这也是为什么新脚本把 YouTube 下载做成显式 `--download-youtube`，并在失败样本上继续扫描。

### 2. 备选：`HuggingFaceM4/ActivitiyNet_Captions`

适合需要更长动作片段时使用：

- 每个视频有 temporal caption start/end，可裁剪为 10-40 秒片段。
- caption 平均约十几个词，需要截短到 8 个词以内。
- 视频源可用性比直接托管 MP4 更不稳定，因此建议只作为第二阶段扩充。

### 3. 直接托管 MP4 fallback

如果环境不能使用 `yt-dlp`，可以临时用直接托管在 Hugging Face 的小型 video-caption 数据集验证管线，例如 `Databoost/VidData` 或 `ShareGPT4Video/ShareGPT4Video`。它们更容易 materialize，但 caption 往往更长、片段可能更短，所以只能作为 smoke/ablation，不建议作为最终实验主数据源。

## 运行方案

### A. 构建 MSR-VTT 短文本 manifest

```bash
python scripts/build_hf_video_text_dataset.py \
  --dataset VLM2Vec/MSR-VTT \
  --config train_7k \
  --split train \
  --output-dir Data/msrvtt-short-text \
  --target-rows 500 \
  --scan-rows 5000 \
  --min-duration 4 \
  --max-duration 40 \
  --max-caption-words 8 \
  --download-youtube
```

Windows PowerShell 里不要用 `\` 续行，应该用反引号：

```powershell
python scripts/build_hf_video_text_dataset.py `
  --dataset VLM2Vec/MSR-VTT `
  --config train_7k `
  --split train `
  --output-dir Data/msrvtt-short-text `
  --target-rows 500 `
  --scan-rows 5000 `
  --min-duration 4 `
  --max-duration 40 `
  --max-caption-words 8 `
  --download-youtube
```

如果是在 Windows CMD，用 `^`：

```bat
python scripts/build_hf_video_text_dataset.py ^
  --dataset VLM2Vec/MSR-VTT ^
  --config train_7k ^
  --split train ^
  --output-dir Data/msrvtt-short-text ^
  --target-rows 500 ^
  --scan-rows 5000 ^
  --min-duration 4 ^
  --max-duration 40 ^
  --max-caption-words 8 ^
  --download-youtube
```

输出文件：

- `Data/msrvtt-short-text/manifest.csv`：训练脚本直接使用的 `video_path,text`。
- `Data/msrvtt-short-text/manifest_with_sanity.csv`：带 duration、fps、sha256、source row 的可审计 manifest。
- `Data/msrvtt-short-text/rejected.csv`：下载失败、时长不合格、重复视频/文本等被拒原因。
- `Data/msrvtt-short-text/sanity_report.json`：质量闸门结果。

### B. 用新 manifest 训练

```bash
python model/qcl_train_refactored.py \
  --task video_text \
  --manifest Data/msrvtt-short-text/manifest.csv \
  --feature-backend video_encoder \
  --video-encoder-model microsoft/xclip-base-patch32 \
  --num-frames 16 \
  --text-pooling truncate \
  --text-preprocess first_n_words \
  --text-first-n-words 8 \
  --result-path results/msrvtt_xclip16_raw \
  --model-path results/msrvtt_xclip16_raw_models
```

建议先跑 raw adapter 作为数据质量基线；只有 raw baseline 明显高于随机检索时，再比较 quantum adapter。

## Sanity check 闸门

新脚本会在写 `manifest.csv` 前执行以下检查：

1. **OpenCV 可解码**：无法打开、帧数为 0、FPS 为 0 的视频拒绝。
2. **时长过滤**：默认保留 4-40 秒，避免极短 GIF 或长视频污染 batch。
3. **文本长度过滤**：默认保留至少 2 个词，最多截到 8 个词。
4. **caption 去重**：规范化后的 caption 完全相同则拒绝后续样本。
5. **exact video 去重**：SHA256 完全相同则拒绝。
6. **near-duplicate video 去重**：对 5 个均匀采样帧做 8x8 average-hash，拼接后用 Hamming distance 过滤近重复。
7. **整体质量阈值**：默认要求 caption unique rate ≥ 0.85，duplicate reject rate ≤ 0.05，且 accepted 数达到 `--target-rows`；否则脚本以非 0 状态退出。

## 训练前必须看的三个数

打开 `sanity_report.json`，确认：

- `accepted == target_rows`：没有因为下载/过滤导致样本不足。
- `caption_unique_rate >= 0.85`：文本不是大量重复模板。
- `duplicate_reject_rate <= 0.05`：视频近重复没有严重到污染训练。

如果这里失败，不要训练；先换数据源、增大 `--scan-rows`，或放宽/收紧时长与重复阈值。
