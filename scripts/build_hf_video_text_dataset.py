#!/usr/bin/env python3
"""Build and sanity-check a compact Hugging Face video-description dataset.

The output manifest is compatible with model/video_text_trainer.py:

    video_path,text
    videos/sample_000001.mp4,a person kicks a ball

Recommended first pass on Bash:

    python scripts/build_hf_video_text_dataset.py \
      --dataset VLM2Vec/MSR-VTT --config train_7k --split train \
      --output-dir Data/msrvtt-short-text --target-rows 500 \
      --download-youtube

On PowerShell, use backticks instead of backslashes:

    python scripts/build_hf_video_text_dataset.py `
      --dataset VLM2Vec/MSR-VTT --config train_7k --split train `
      --output-dir Data/msrvtt-short-text --target-rows 500 `
      --download-youtube
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "Data" / "hf-video-text-short"
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize a high-quality short video-description dataset from Hugging Face."
    )
    parser.add_argument("--dataset", default="VLM2Vec/MSR-VTT", help="Hugging Face dataset repo id.")
    parser.add_argument("--config", default="train_7k", help="Dataset config/subset; pass '' when unused.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output dataset directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--target-rows", type=int, default=500, help="Number of accepted pairs to write.")
    parser.add_argument("--scan-rows", type=int, default=5000, help="Rows to scan before giving up.")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Minimum decoded video duration in seconds.")
    parser.add_argument("--max-duration", type=float, default=40.0, help="Maximum decoded video duration in seconds.")
    parser.add_argument("--max-caption-words", type=int, default=8, help="Keep only the first N caption words.")
    parser.add_argument("--min-caption-words", type=int, default=2, help="Drop captions shorter than this.")
    parser.add_argument("--video-column", default=None, help="Override the video/path column.")
    parser.add_argument("--text-column", default=None, help="Override the caption/text column.")
    parser.add_argument("--url-column", default=None, help="Override URL column for YouTube/direct source datasets.")
    parser.add_argument("--start-column", default=None, help="Override clip start-time column.")
    parser.add_argument("--end-column", default=None, help="Override clip end-time column.")
    parser.add_argument("--duration-column", default=None, help="Override duration column.")
    parser.add_argument(
        "--preset",
        choices=["generic", "finevideo"],
        default="generic",
        help="Dataset-specific extraction preset.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode. Recommended for large video datasets such as FineVideo.",
    )
    parser.add_argument(
        "--streaming-shuffle-buffer",
        type=int,
        default=1000,
        help="Shuffle buffer for streaming datasets; set 0 to keep repository order.",
    )
    parser.add_argument(
        "--finevideo-category",
        default=None,
        help="Optional comma-separated FineVideo content_parent_category filter, e.g. Sports,Education.",
    )
    parser.add_argument(
        "--download-youtube",
        action="store_true",
        help="Use yt-dlp for rows whose only source is a YouTube URL. Requires yt-dlp in PATH.",
    )
    parser.add_argument("--max-near-duplicate-rate", type=float, default=0.05)
    parser.add_argument("--min-caption-unique-rate", type=float, default=0.85)
    parser.add_argument("--near-duplicate-hamming", type=int, default=18)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=180,
        help="Per-video direct/yt-dlp download timeout in seconds.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print a scan progress line every N candidate rows; 1 logs every candidate.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        if any(token == "\\" for token in unknown):
            parser.error(
                "found literal '\\' argument(s). On Windows, use PowerShell backtick (`) "
                "or CMD caret (^) for line continuation, or put the command on one line; "
                "Unix backslash continuations are passed as arguments."
            )
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def log(message):
    print(message, flush=True)


def first_existing_column(columns, preferred):
    lower_to_original = {name.lower(): name for name in columns}
    for name in preferred:
        if name in columns:
            return name
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    return None


def infer_columns(dataset, args):
    columns = dataset.column_names or []
    if not columns and getattr(dataset, "features", None):
        columns = list(dataset.features)
    if args.preset == "finevideo" and not columns:
        columns = ["mp4", "json"]
    if args.preset == "finevideo":
        video_column = args.video_column or first_existing_column(columns, ["mp4", "video"])
        text_column = args.text_column or first_existing_column(columns, ["json"])
    else:
        video_column = args.video_column or first_existing_column(
            columns, ["video", "videos", "video_path", "video_name", "file_name", "path", "mp4"]
        )
        text_column = args.text_column or first_existing_column(
            columns, ["caption", "text", "description", "sentence", "summary", "Caption"]
        )
    url_column = args.url_column or first_existing_column(columns, ["url", "youtube_url", "video_url", "download_url"])
    start_column = args.start_column or first_existing_column(columns, ["start", "start time", "start_time", "clip_start"])
    end_column = args.end_column or first_existing_column(columns, ["end", "end time", "end_time", "clip_end"])
    duration_column = args.duration_column or first_existing_column(columns, ["duration", "duration_seconds", "seconds"])

    if not text_column:
        raise ValueError(f"No caption/text column found. Columns: {columns}")
    if not video_column and not url_column:
        raise ValueError(f"No video/path/url column found. Columns: {columns}")
    return video_column, text_column, url_column, start_column, end_column, duration_column


def caption_candidates(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text[:1] in "[(":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item).strip()]
    return [text]


def normalize_caption(text, max_words):
    text = str(text or "").replace("\n", " ").strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    words = text.split()
    return " ".join(words[:max_words])


def clean_caption(value, min_words, max_words):
    cleaned = [normalize_caption(candidate, max_words) for candidate in caption_candidates(value)]
    cleaned = [caption for caption in cleaned if caption]
    valid = [caption for caption in cleaned if len(caption.split()) >= min_words]
    if valid:
        return max(valid, key=lambda caption: len(caption.split()))
    if cleaned:
        return max(cleaned, key=lambda caption: len(caption.split()))
    return ""


def parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    hours, minutes, seconds = values
    return hours * 3600 + minutes * 60 + seconds


def timestamp_range(value):
    if isinstance(value, dict):
        start = parse_timestamp(value.get("start_timestamp") or value.get("start") or value.get("start_time"))
        end = parse_timestamp(value.get("end_timestamp") or value.get("end") or value.get("end_time"))
        return start, end
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    start = parse_timestamp(value[0])
    end = parse_timestamp(value[1])
    return start, end


def finevideo_category_allowed(metadata, category_filter):
    if not category_filter:
        return True
    allowed = {part.strip().lower() for part in category_filter.split(",") if part.strip()}
    category = str(metadata.get("content_parent_category") or "").strip().lower()
    return category in allowed


def finevideo_clip_candidates(metadata, args):
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            try:
                metadata = ast.literal_eval(metadata)
            except (SyntaxError, ValueError):
                metadata = {}
    if not isinstance(metadata, dict) or not finevideo_category_allowed(metadata, args.finevideo_category):
        return []

    content = metadata.get("content_metadata") or {}
    candidates = []

    for suggestion_index, suggestion in enumerate(content.get("trimmingSuggestions") or []):
        if not isinstance(suggestion, dict):
            continue
        start, end = timestamp_range(suggestion.get("timestamps") or suggestion.get("timestamp"))
        if start is None or end is None or end <= start:
            continue
        duration = end - start
        if duration < args.min_duration or duration > args.max_duration:
            continue
        caption = clean_caption(
            suggestion.get("description") or suggestion.get("title"),
            args.min_caption_words,
            args.max_caption_words,
        )
        if len(caption.split()) < args.min_caption_words:
            continue
        candidates.append(
            {
                "caption": caption,
                "clip_start": round(start, 3),
                "clip_end": round(end, 3),
                "clip_kind": "finevideo-trimming-suggestion",
                "scene_index": None,
                "activity_index": suggestion_index,
                "category": metadata.get("content_parent_category"),
            }
        )

    scenes = content.get("scenes") or []
    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        activities = scene.get("activities") or []
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict):
                continue
            start, end = timestamp_range(activity.get("timestamps") or activity.get("timestamp"))
            if start is None or end is None or end <= start:
                continue
            duration = end - start
            if duration < args.min_duration or duration > args.max_duration:
                continue
            caption = clean_caption(
                activity.get("description") or activity.get("title"),
                args.min_caption_words,
                args.max_caption_words,
            )
            if len(caption.split()) < args.min_caption_words:
                continue
            candidates.append(
                {
                    "caption": caption,
                    "clip_start": round(start, 3),
                    "clip_end": round(end, 3),
                    "clip_kind": "finevideo-activity",
                    "scene_index": scene_index,
                    "activity_index": activity_index,
                    "category": metadata.get("content_parent_category"),
                }
            )
    return candidates


def choose_finevideo_clip(row, text_column, args, rng):
    candidates = finevideo_clip_candidates(row.get(text_column), args)
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def parse_hf_resolve_url(value):
    parsed = urlparse(value)
    if parsed.netloc != "huggingface.co":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 6 or parts[0] != "datasets" or "resolve" not in parts:
        return None
    resolve_index = parts.index("resolve")
    repo_id = "/".join(parts[1:resolve_index])
    revision = parts[resolve_index + 1]
    filename = "/".join(parts[resolve_index + 2:])
    return repo_id, revision, filename


def download_direct_url(url, output_path, timeout):
    parsed_hf = parse_hf_resolve_url(url)
    if parsed_hf:
        repo_id, revision, filename = parsed_hf
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
        shutil.copy2(downloaded, output_path)
        return "hf-resolve-url"
    with urllib.request.urlopen(url, timeout=timeout) as response, Path(output_path).open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return "url"


def materialize_hf_path(dataset_name, config, source_path, output_path, cache_dir):
    source_path = str(source_path).replace("\\", "/").lstrip("/")
    candidates = [source_path]
    if config:
        candidates.extend([f"{config}/{source_path}", f"videos/{source_path}"])
    for candidate in dict.fromkeys(candidates):
        downloaded = hf_hub_download(
            repo_id=dataset_name,
            filename=candidate,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        if Path(downloaded).exists():
            shutil.copy2(downloaded, output_path)
            return f"hf-file:{candidate}"
    raise FileNotFoundError(source_path)


def run_youtube_download(url, output_path, start_time, end_time, timeout):
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not installed; rerun without YouTube sources or install yt-dlp")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_template = str(Path(tmpdir) / "clip.%(ext)s")
        command = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--socket-timeout",
            "20",
            "--retries",
            "2",
            "--fragment-retries",
            "2",
            "--force-keyframes-at-cuts",
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
            "-o",
            tmp_template,
        ]
        if start_time is not None and end_time is not None:
            command.extend(["--download-sections", f"*{float(start_time)}-{float(end_time)}"])
        command.append(url)
        try:
            subprocess.run(
                command,
                check=True,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"yt-dlp timed out after {timeout}s") from exc
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "").strip().splitlines()
            detail = " | ".join(output[-3:]) if output else f"exit code {exc.returncode}"
            raise RuntimeError(f"yt-dlp failed: {detail}") from exc
        candidates = sorted(Path(tmpdir).glob("clip.*"))
        if not candidates:
            raise RuntimeError("yt-dlp produced no output file")
        shutil.copy2(candidates[0], output_path)
    return "yt-dlp"


def row_source(row, video_column, url_column):
    value = row.get(video_column) if video_column else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes", bytes(value)
    if isinstance(value, dict):
        if value.get("bytes"):
            return "bytes", value
        if value.get("path"):
            return "path", value["path"]
    if value:
        return "path", str(value)
    if url_column and row.get(url_column):
        return "url", str(row[url_column])
    return None, None


def materialize_unclipped_row(row, args, output_path, video_column, url_column, start_column, end_column):
    source_kind, source_value = row_source(row, video_column, url_column)
    if source_kind == "bytes":
        payload = source_value.get("bytes") if isinstance(source_value, dict) else source_value
        output_path.write_bytes(payload)
        return "embedded-bytes"
    if not source_value:
        raise RuntimeError("no source")

    source = str(source_value)
    if source.startswith(("http://", "https://")):
        if "youtube.com/" in source or "youtu.be/" in source:
            if not args.download_youtube:
                raise RuntimeError("youtube source skipped because --download-youtube was not set")
            start = row.get(start_column) if start_column else None
            end = row.get(end_column) if end_column else None
            return run_youtube_download(source, output_path, start, end, args.download_timeout)
        return download_direct_url(source, output_path, args.download_timeout)

    local_path = Path(source)
    if local_path.exists():
        shutil.copy2(local_path, output_path)
        return "local-path"

    try:
        return materialize_hf_path(args.dataset, args.config or None, source, output_path, args.cache_dir)
    except Exception:
        if url_column and row.get(url_column) and str(row[url_column]) != source:
            fallback_url = str(row[url_column])
            if "youtube.com/" in fallback_url or "youtu.be/" in fallback_url:
                if not args.download_youtube:
                    raise RuntimeError("youtube fallback skipped because --download-youtube was not set")
                start = row.get(start_column) if start_column else None
                end = row.get(end_column) if end_column else None
                return run_youtube_download(fallback_url, output_path, start, end, args.download_timeout)
            return download_direct_url(fallback_url, output_path, args.download_timeout)
        raise


def clip_video_opencv(input_path, output_path, start_time, end_time):
    if start_time is None or end_time is None or end_time <= start_time:
        raise RuntimeError(f"invalid clip range: {start_time}-{end_time}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV cannot open source video for clipping")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("source video has invalid FPS, size, or frame count")

    start_frame = max(0, int(round(start_time * fps)))
    end_frame = min(frame_count, int(round(end_time * fps)))
    if end_frame <= start_frame:
        capture.release()
        raise RuntimeError(f"clip frame range is empty: {start_frame}-{end_frame}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV cannot open output video writer")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(start_frame, end_frame):
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    writer.release()
    capture.release()
    if written <= 0:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError("OpenCV clip writer produced no frames")


def materialize_row(row, args, output_path, video_column, url_column, start_column, end_column, clip_start=None, clip_end=None):
    if clip_start is None or clip_end is None:
        return materialize_unclipped_row(row, args, output_path, video_column, url_column, start_column, end_column)
    with tempfile.TemporaryDirectory() as tmpdir:
        full_path = Path(tmpdir) / "source.mp4"
        source = materialize_unclipped_row(row, args, full_path, video_column, url_column, start_column, end_column)
        clip_video_opencv(full_path, output_path, float(clip_start), float(clip_end))
    return f"{source}|opencv-clip:{float(clip_start):.3f}-{float(clip_end):.3f}"


def video_stats(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV cannot open video")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    capture.release()
    if frames <= 0 or duration <= 0:
        raise RuntimeError("empty or undecodable video")
    return {"frames": frames, "fps": fps, "duration": duration, "width": width, "height": height}


def frame_average_hash(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8).flatten()


def video_fingerprint(path, samples=5):
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("cannot fingerprint empty video")
    positions = np.linspace(0, max(frame_count - 1, 0), samples).astype(int)
    hashes = []
    for pos in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = capture.read()
        if ok:
            hashes.append(frame_average_hash(frame))
    capture.release()
    if not hashes:
        raise RuntimeError("cannot decode fingerprint frames")
    return np.concatenate(hashes)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset.lower() == "huggingfacefv/finevideo" and args.preset == "generic":
        log("Detected HuggingFaceFV/finevideo; using --preset finevideo.")
        args.preset = "finevideo"
    if args.preset == "finevideo":
        if args.config == "train_7k":
            args.config = ""
        if not args.streaming:
            log("FineVideo is large and gated; enabling Hugging Face streaming mode.")
            args.streaming = True

    log(f"Loading dataset={args.dataset} config={args.config or '<none>'} split={args.split}")
    try:
        dataset = load_dataset(
            args.dataset,
            args.config or None,
            split=args.split,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
    except Exception as exc:
        if args.preset == "finevideo":
            raise SystemExit(
                "Could not load HuggingFaceFV/finevideo. It is a gated dataset: accept the terms "
                "on Hugging Face and authenticate locally with `huggingface-cli login` or HF_TOKEN."
            ) from exc
        raise
    if args.streaming and args.streaming_shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.streaming_shuffle_buffer, seed=args.seed)
    video_column, text_column, url_column, start_column, end_column, duration_column = infer_columns(dataset, args)
    log(
        "Columns: "
        f"video={video_column} text={text_column} url={url_column} "
        f"start={start_column} end={end_column} duration={duration_column}"
    )
    if args.download_youtube and shutil.which("yt-dlp") is None:
        raise SystemExit(
            "This dataset requires YouTube downloads, but yt-dlp is not installed. "
            "Install it with `python -m pip install -U yt-dlp`, then rerun this command; "
            "or switch to a dataset that hosts video files directly."
        )

    if args.streaming:
        scan_limit = max(args.scan_rows, args.target_rows)
        row_iter = enumerate(dataset)
    else:
        scan_limit = min(len(dataset), max(args.scan_rows, args.target_rows))
        indices = np.arange(scan_limit)
        rng.shuffle(indices)
        row_iter = ((int(row_index), dataset[int(row_index)]) for row_index in indices)
    log(
        f"Scanning up to {scan_limit} rows for {args.target_rows} accepted examples "
        f"(download timeout: {args.download_timeout}s/video)"
    )

    accepted = []
    rejected = []
    fingerprints = []
    caption_counts = {}
    exact_hashes = set()

    for scanned_count, (row_index, row) in enumerate(row_iter, start=1):
        if scanned_count > scan_limit:
            break
        if len(accepted) >= args.target_rows:
            break
        if args.log_every > 0 and (scanned_count == 1 or scanned_count % args.log_every == 0):
            log(f"scan {scanned_count:>5}/{scan_limit}: accepted={len(accepted)} rejected={len(rejected)}")
        clip_info = None
        if args.preset == "finevideo":
            clip_info = choose_finevideo_clip(row, text_column, args, rng)
            if not clip_info:
                rejected.append({"row": int(row_index), "reason": "no-finevideo-short-clip", "detail": "no activity candidate matched filters"})
                continue
            caption = clip_info["caption"]
        else:
            caption = clean_caption(row.get(text_column), args.min_caption_words, args.max_caption_words)
        word_count = len(caption.split())
        if word_count < args.min_caption_words:
            rejected.append({"row": int(row_index), "reason": "caption-too-short", "detail": caption})
            continue
        if caption_counts.get(caption, 0) > 0:
            rejected.append({"row": int(row_index), "reason": "duplicate-caption", "detail": caption})
            continue

        suffix = ".mp4"
        source_kind, source_value = row_source(row, video_column, url_column)
        if source_value:
            suffix_candidate = Path(str(source_value)).suffix.lower()
            if suffix_candidate in VIDEO_SUFFIXES:
                suffix = suffix_candidate
        output_path = videos_dir / f"sample_{len(accepted):06d}{suffix}"
        log(
            f"trying row={int(row_index)} accepted={len(accepted)}/{args.target_rows} "
            f"source={source_kind or 'none'} caption={caption!r}"
        )

        try:
            clip_start = clip_info["clip_start"] if clip_info else None
            clip_end = clip_info["clip_end"] if clip_info else None
            source = materialize_row(
                row,
                args,
                output_path,
                video_column,
                url_column,
                start_column,
                end_column,
                clip_start=clip_start,
                clip_end=clip_end,
            )
            stats = video_stats(output_path)
            duration_hint = float(row[duration_column]) if duration_column and row.get(duration_column) else stats["duration"]
            if duration_hint < args.min_duration or duration_hint > args.max_duration:
                raise RuntimeError(f"duration-out-of-range:{duration_hint:.2f}s")
            digest = file_sha256(output_path)
            if digest in exact_hashes:
                raise RuntimeError("exact-video-duplicate")
            fingerprint = video_fingerprint(output_path)
            if any(hamming(fingerprint, prior) <= args.near_duplicate_hamming for prior in fingerprints):
                raise RuntimeError("near-video-duplicate")
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            rejected.append({"row": int(row_index), "reason": "materialize-or-sanity-failed", "detail": str(exc)[:300]})
            log(f"  skip row={int(row_index)}: {str(exc)[:220]}")
            continue

        exact_hashes.add(digest)
        fingerprints.append(fingerprint)
        caption_counts[caption] = caption_counts.get(caption, 0) + 1
        accepted.append(
            {
                "video_path": str(output_path.relative_to(output_dir)),
                "text": caption,
                "source_row": int(row_index),
                "source": source,
                "duration": round(stats["duration"], 3),
                "frames": stats["frames"],
                "fps": round(stats["fps"], 3),
                "width": stats["width"],
                "height": stats["height"],
                "sha256": digest,
            }
        )
        if clip_info:
            accepted[-1].update(
                {
                    "clip_start": clip_info["clip_start"],
                    "clip_end": clip_info["clip_end"],
                    "clip_kind": clip_info["clip_kind"],
                    "category": clip_info.get("category"),
                    "scene_index": clip_info.get("scene_index"),
                    "activity_index": clip_info.get("activity_index"),
                }
            )
        log(f"accepted {len(accepted):>5}/{args.target_rows}: row={row_index} {caption!r}")

    manifest_rows = [{"video_path": row["video_path"], "text": row["text"]} for row in accepted]
    write_csv(output_dir / "manifest.csv", manifest_rows, ["video_path", "text"])
    write_csv(output_dir / "manifest_with_sanity.csv", accepted, list(accepted[0].keys()) if accepted else ["video_path", "text"])
    write_csv(output_dir / "rejected.csv", rejected, ["row", "reason", "detail"])

    duplicate_rejects = sum("duplicate" in row["detail"] or "duplicate" in row["reason"] for row in rejected)
    scanned = len(accepted) + len(rejected)
    caption_unique_rate = len(caption_counts) / max(len(accepted), 1)
    near_duplicate_rate = duplicate_rejects / max(scanned, 1)
    durations = [row["duration"] for row in accepted]
    report = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "target_rows": args.target_rows,
        "caption_unique_rate": round(caption_unique_rate, 4),
        "duplicate_reject_rate": round(near_duplicate_rate, 4),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "mean": round(float(np.mean(durations)), 3) if durations else None,
            "max": max(durations) if durations else None,
        },
        "pass": bool(
            len(accepted) >= args.target_rows
            and caption_unique_rate >= args.min_caption_unique_rate
            and near_duplicate_rate <= args.max_near_duplicate_rate
        ),
        "thresholds": {
            "min_caption_unique_rate": args.min_caption_unique_rate,
            "max_near_duplicate_rate": args.max_near_duplicate_rate,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
        },
    }
    (output_dir / "sanity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Sanity check failed; inspect sanity_report.json and rejected.csv before training.")


if __name__ == "__main__":
    main()
