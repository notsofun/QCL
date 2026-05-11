#!/usr/bin/env python3
"""Curate a short, low-duplicate Hugging Face video-description dataset.

The default target is VLM2Vec/MSR-VTT because it is a compact, public
text-to-video retrieval benchmark with short clips and human captions.  The
script intentionally writes the same `manifest.csv` schema that the existing
video_text trainer consumes: `video_path,text`.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data" / "msrvtt-short-video-description"
VIDEO_COLUMN_CANDIDATES = ("video", "videos", "video_path", "path", "mp4", "file")
TEXT_COLUMN_CANDIDATES = ("caption", "text", "description", "sentence", "sentences")
BANNED_TEXT_PATTERNS = (
    # Keep the default conservative: remove obvious adult/nudity captions that can
    # create governance or review friction in quick experiments.
    r"\bnaked\b",
    r"\bnude\b",
    r"\bporn\w*\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/filter a high-quality short video-description dataset from Hugging Face."
    )
    parser.add_argument("--dataset", default="VLM2Vec/MSR-VTT")
    parser.add_argument("--config", default="test_1k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--scan-rows", type=int, default=2500)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=40.0)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if sanity checks warn about duplicates, decode failures, or bad lengths.",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Skip Hugging Face download and only validate an existing manifest in --output-dir.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def normalise_text(text: Any) -> str:
    text = str(text).replace("<video>", " ").replace("<image>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Prefer one concise clause over long generated paragraphs.
    text = re.split(r"[.;\n]", text)[0].strip(" ,")
    return text


def text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def has_banned_text(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in BANNED_TEXT_PATTERNS)


def infer_column(columns: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    for column in columns:
        lowered = column.lower()
        if any(candidate in lowered for candidate in candidates):
            return column
    raise ValueError(f"Could not infer {kind} column from columns: {list(columns)}")


def row_duration_seconds(row: dict[str, Any]) -> float | None:
    for start_name, end_name in (("start time", "end time"), ("start", "end"), ("start_time", "end_time")):
        if start_name in row and end_name in row:
            try:
                return float(row[end_name]) - float(row[start_name])
            except (TypeError, ValueError):
                return None
    for name in ("duration", "duration_seconds", "seconds"):
        if name in row:
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return None
    return None


def hf_candidate_paths(source_path: str, config: str | None) -> list[str]:
    source_path = source_path.replace("\\", "/").lstrip("/")
    basename = Path(source_path).name
    prefixes = ["", "videos", "video", "data", "raw_data"]
    if config:
        prefixes.extend([config, f"{config}/videos", f"{config}/video", f"{config}/data"])
    candidates = [source_path, basename]
    candidates.extend(f"{prefix}/{source_path}" for prefix in prefixes if prefix)
    candidates.extend(f"{prefix}/{basename}" for prefix in prefixes if prefix)
    return list(dict.fromkeys(candidates))


def copy_video_asset(video_value: Any, dataset_name: str, config: str | None, output_path: Path, cache_dir: str | None) -> str:
    source_path: str | None = None
    if isinstance(video_value, dict):
        if video_value.get("bytes"):
            output_path.write_bytes(video_value["bytes"])
            return "embedded-bytes"
        source_path = video_value.get("path") or video_value.get("filename")
    elif isinstance(video_value, (str, os.PathLike)):
        source_path = str(video_value)

    if not source_path:
        raise RuntimeError("row has no usable video path/bytes")

    source = Path(source_path)
    if source.exists():
        shutil.copy2(source, output_path)
        return f"local:{source}"

    from huggingface_hub import hf_hub_download

    errors: list[str] = []
    for candidate in hf_candidate_paths(source_path, config):
        try:
            downloaded = hf_hub_download(
                repo_id=dataset_name,
                filename=candidate,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 - report the last few Hub path failures to the user.
            errors.append(f"{candidate}: {exc}")
            continue
        shutil.copy2(downloaded, output_path)
        return f"hub:{candidate}"

    raise RuntimeError("could not download video asset; last errors: " + " | ".join(errors[-3:]))


def video_stats(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"decoded": False, "frames": 0, "fps": 0.0, "duration": 0.0, "frame_hash": None}

    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = frames / fps if fps > 0 else 0.0
    sampled_hashes = []
    if frames > 0:
        for frame_index in np.linspace(0, max(frames - 1, 0), num=min(8, frames), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (16, 16)), cv2.COLOR_BGR2GRAY)
            sampled_hashes.append("".join("1" if px > gray.mean() else "0" for px in gray.flatten()))
    capture.release()
    digest = hashlib.sha1("".join(sampled_hashes).encode("ascii")).hexdigest() if sampled_hashes else None
    return {"decoded": True, "frames": frames, "fps": fps, "duration": duration, "frame_hash": digest}


def file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanity_check(output_dir: Path, strict: bool = False) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    diagnostics: dict[str, Any] = {
        "manifest": str(manifest_path),
        "num_rows": len(rows),
        "warnings": [],
        "duration_seconds": {},
        "word_counts": {},
        "duplicate_text_groups": [],
        "duplicate_file_hash_groups": [],
        "duplicate_frame_hash_groups": [],
        "decode_failures": [],
    }

    text_groups: dict[str, list[int]] = defaultdict(list)
    file_hash_groups: dict[str, list[int]] = defaultdict(list)
    frame_hash_groups: dict[str, list[int]] = defaultdict(list)
    durations: list[float] = []
    words: list[int] = []

    for index, row in enumerate(rows):
        rel_video = row["video_path"]
        text = row["text"]
        words.append(word_count(text))
        text_groups[text_key(text)].append(index)
        path = output_dir / rel_video
        stats = video_stats(path)
        if not stats["decoded"]:
            diagnostics["decode_failures"].append({"row": index, "video_path": rel_video})
            continue
        durations.append(float(stats["duration"]))
        file_hash_groups[file_sha1(path)].append(index)
        if stats["frame_hash"]:
            frame_hash_groups[str(stats["frame_hash"])].append(index)

    def duplicate_groups(groups: dict[str, list[int]]) -> list[list[int]]:
        return [indices for indices in groups.values() if len(indices) > 1]

    diagnostics["duplicate_text_groups"] = duplicate_groups(text_groups)
    diagnostics["duplicate_file_hash_groups"] = duplicate_groups(file_hash_groups)
    diagnostics["duplicate_frame_hash_groups"] = duplicate_groups(frame_hash_groups)
    if durations:
        diagnostics["duration_seconds"] = {
            "min": min(durations),
            "median": float(np.median(durations)),
            "max": max(durations),
        }
    if words:
        diagnostics["word_counts"] = {"min": min(words), "median": float(np.median(words)), "max": max(words)}

    if diagnostics["decode_failures"]:
        diagnostics["warnings"].append("some videos cannot be decoded")
    if diagnostics["duplicate_text_groups"]:
        diagnostics["warnings"].append("duplicate captions found")
    if diagnostics["duplicate_file_hash_groups"] or diagnostics["duplicate_frame_hash_groups"]:
        diagnostics["warnings"].append("duplicate or near-duplicate videos found")
    if len(rows) < 32:
        diagnostics["warnings"].append("dataset is tiny; retrieval metrics will be noisy")

    report_path = output_dir / "sanity_report.json"
    report_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Sanity report: {report_path}")
    for warning in diagnostics["warnings"]:
        log(f"WARNING: {warning}")
    if strict and diagnostics["warnings"]:
        raise SystemExit(2)
    return diagnostics


def curate(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading dataset={args.dataset} config={args.config or '<none>'} split={args.split}")
    config = args.config or None
    if config:
        dataset = load_dataset(args.dataset, config, split=args.split, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    video_column = infer_column(dataset.column_names, VIDEO_COLUMN_CANDIDATES, "video")
    text_column = infer_column(dataset.column_names, TEXT_COLUMN_CANDIDATES, "text")
    log(f"Using columns: video={video_column!r}, text={text_column!r}; rows={len(dataset)}")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    indices = indices[: min(args.scan_rows, len(indices))]

    manifest_rows: list[dict[str, str]] = []
    seen_text: set[str] = set()
    seen_video_ids: set[str] = set()
    skip_reasons: Counter[str] = Counter()

    for row_index in indices:
        if len(manifest_rows) >= args.max_samples:
            break
        row = dataset[int(row_index)]
        text = normalise_text(row[text_column])
        words = word_count(text)
        if words < args.min_words or words > args.max_words:
            skip_reasons["caption_length"] += 1
            continue
        if has_banned_text(text):
            skip_reasons["banned_text"] += 1
            continue
        key = text_key(text)
        if key in seen_text:
            skip_reasons["duplicate_text"] += 1
            continue
        duration = row_duration_seconds(row)
        if duration is not None and not (args.min_duration <= duration <= args.max_duration):
            skip_reasons["metadata_duration"] += 1
            continue
        video_id = str(row.get("video_id") or row.get("id") or row[video_column])
        if video_id in seen_video_ids:
            skip_reasons["duplicate_video_id"] += 1
            continue

        suffix = Path(str(row[video_column].get("path", "") if isinstance(row[video_column], dict) else row[video_column])).suffix or ".mp4"
        video_path = videos_dir / f"sample_{len(manifest_rows):05d}{suffix}"
        try:
            source = copy_video_asset(row[video_column], args.dataset, config, video_path, args.cache_dir)
            stats = video_stats(video_path)
        except Exception as exc:  # noqa: BLE001 - continue scanning and surface counts.
            skip_reasons["download_or_decode_exception"] += 1
            log(f"  skip row {row_index}: {exc}")
            video_path.unlink(missing_ok=True)
            continue
        if not stats["decoded"] or stats["frames"] <= 0:
            skip_reasons["decode_failure"] += 1
            video_path.unlink(missing_ok=True)
            continue
        actual_duration = float(stats["duration"])
        if actual_duration and not (args.min_duration <= actual_duration <= args.max_duration):
            skip_reasons["actual_duration"] += 1
            video_path.unlink(missing_ok=True)
            continue

        seen_text.add(key)
        seen_video_ids.add(video_id)
        manifest_rows.append({"video_path": str(video_path.relative_to(output_dir)), "text": text})
        log(f"  keep {len(manifest_rows):04d}: row={row_index} {actual_duration:.1f}s words={words} source={source}")

    if not manifest_rows:
        raise RuntimeError(f"No valid samples collected. Skip reasons: {dict(skip_reasons)}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "dataset": args.dataset,
        "config": config,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "scan_rows": args.scan_rows,
        "filters": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "min_words": args.min_words,
            "max_words": args.max_words,
        },
        "skip_reasons": dict(skip_reasons),
    }
    (output_dir / "curation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not args.sanity_only:
        curate(args)
    sanity_check(output_dir, strict=args.strict)


if __name__ == "__main__":
    main()
