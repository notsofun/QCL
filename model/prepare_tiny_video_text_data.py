import argparse
import csv
import ast
import json
import os
import re
import shutil
import tarfile
from urllib.parse import urlparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "Data" / "tiny-video-text"


parser = argparse.ArgumentParser(description="Materialize a tiny video-text dataset into local files.")
parser.add_argument("--dataset", default="lmms-lab/LLaVA-Video-178K", type=str)
parser.add_argument("--config", default="0_30_s_academic_v0_1", type=str,
                    help="Hugging Face dataset config/subset name. Use an empty string to disable.")
parser.add_argument("--split", default="caption", type=str)
parser.add_argument("--max-rows", default=8, type=int)
parser.add_argument("--scan-rows", default=50, type=int,
                    help="Rows to inspect while collecting max-rows valid videos.")
parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
parser.add_argument("--cache-dir", default=None, type=str)
parser.add_argument("--llava-archive", default=None, type=str,
                    help="Local LLaVA video tar.gz archive, or comma-separated archives, to extract videos from.")
parser.add_argument("--download-llava-shard", default=None, type=int,
                    help="Download one LLaVA video archive shard, e.g. 8 for the smaller videos_8 tar.gz.")
parser.add_argument(
    "--hf",
    action="store_true",
    help="Download/materialize the Hugging Face dataset instead of creating local synthetic videos.",
)
parser.add_argument(
    "--synthetic",
    action="store_true",
    help="Create local synthetic videos without network access. This is the default.",
)


def log(message):
    print(message, flush=True)


def parse_hf_dataset_path(path):
    match = re.match(r"^hf://datasets/([^/]+/[^/@]+)(?:@([^/]+))?/(.+)$", path)
    if not match:
        return None
    repo_id, revision, filename = match.groups()
    return repo_id, revision, filename


def parse_hf_resolve_url(url):
    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 6 or parts[0] != "datasets" or "resolve" not in parts:
        return None

    resolve_index = parts.index("resolve")
    if resolve_index < 3 or resolve_index + 2 >= len(parts):
        return None

    repo_id = "/".join(parts[1:resolve_index])
    revision = parts[resolve_index + 1]
    filename = "/".join(parts[resolve_index + 2:])
    return repo_id, revision, filename


def download_hf_file(dataset_name, path, cache_dir):
    from huggingface_hub import hf_hub_download

    parsed = parse_hf_resolve_url(path) or parse_hf_dataset_path(path)
    if parsed:
        repo_id, revision, filename = parsed
    else:
        repo_id, revision, filename = dataset_name, None, path.replace("\\", "/")

    log(f"  downloading video asset from Hub: {filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )


def candidate_hf_paths(source_path, config):
    source_path = source_path.replace("\\", "/").lstrip("/")
    candidates = [source_path]
    if config:
        candidates.extend([
            f"{config}/{source_path}",
            f"LLaVA-Video-178K/{config}/{source_path}",
        ])
    return list(dict.fromkeys(candidates))


def llava_archive_name(config, shard):
    return f"{config}/{config}_videos_{shard}.tar.gz"


def archive_members_for_video(source_path):
    source_path = source_path.replace("\\", "/").lstrip("/")
    basename = Path(source_path).name
    return [
        source_path,
        f"./{source_path}",
        f"LLaVA-Video-178K/{source_path}",
        basename,
        f"./{basename}",
    ]


def extract_video_from_archives(source_path, archive_paths, output_path):
    wanted = set(archive_members_for_video(source_path))
    source_path = source_path.replace("\\", "/").lstrip("/")

    for archive_path in archive_paths:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            continue
        log(f"  searching archive: {archive_path}")
        with tarfile.open(archive_path, "r:*") as archive:
            member = None
            for candidate in archive:
                candidate_name = candidate.name.replace("\\", "/").lstrip("/")
                if (
                    candidate.isfile()
                    and (
                        candidate_name in wanted
                        or candidate_name.endswith("/" + source_path)
                        or candidate_name.endswith("/" + Path(source_path).name)
                    )
                ):
                    member = candidate
                    break

            if member is None:
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with output_path.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            return True, f"archive:{archive_path.name}:{member.name}"

    return False, "not-found-in-archives"


def resolve_llava_archives(args):
    archives = []
    if args.llava_archive:
        archives.extend(Path(path.strip()).resolve() for path in args.llava_archive.split(",") if path.strip())

    if args.download_llava_shard is not None:
        if not args.config:
            raise ValueError("--download-llava-shard requires --config")
        filename = llava_archive_name(args.config, args.download_llava_shard)
        downloaded = download_hf_file(args.dataset, filename, args.cache_dir)
        archives.append(Path(downloaded).resolve())
    return archives


def copy_or_write_video(video_value, dataset_name, output_path, cache_dir, config=None, archive_paths=None):
    source_path = None

    if isinstance(video_value, dict):
        video_bytes = video_value.get("bytes")
        if video_bytes:
            output_path.write_bytes(video_bytes)
            return True, "embedded-bytes"
        source_path = video_value.get("path") or video_value.get("filename")
    elif isinstance(video_value, (str, os.PathLike)):
        source_path = str(video_value)

    if not source_path:
        return False, "no-video-path"

    if archive_paths:
        ok, source = extract_video_from_archives(source_path, archive_paths, output_path)
        if ok:
            return ok, source

    if source_path.startswith(("http://", "https://")):
        hf_resolve_url = parse_hf_resolve_url(source_path)
        if hf_resolve_url:
            downloaded = Path(download_hf_file(dataset_name, source_path, cache_dir))
            if downloaded.exists():
                shutil.copy2(downloaded, output_path)
                return True, "hf-resolve-url"
            return False, "downloaded-hf-url-missing"
        log(f"  downloading video URL: {source_path}")
        urllib.request.urlretrieve(source_path, output_path)
        return True, "url"

    local_path = Path(source_path)
    if local_path.exists():
        shutil.copy2(local_path, output_path)
        return True, "local-path"

    errors = []
    for candidate in candidate_hf_paths(source_path, config):
        try:
            downloaded = Path(download_hf_file(dataset_name, candidate, cache_dir))
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        if downloaded.exists():
            shutil.copy2(downloaded, output_path)
            return True, f"hub-path:{candidate}"

    return False, "hub-download-failed: " + " | ".join(errors[-3:])


def parse_conversations(value):
    if isinstance(value, str):
        for parser_fn in (json.loads, ast.literal_eval):
            try:
                value = parser_fn(value)
                break
            except Exception:
                pass

    if not isinstance(value, list):
        return None

    for message in value:
        if not isinstance(message, dict):
            continue
        speaker = str(message.get("from", "")).lower()
        text = message.get("value")
        if speaker in ("gpt", "assistant") and text:
            return str(text).replace("<image>", "").replace("<video>", "").strip()
    for message in reversed(value):
        if isinstance(message, dict) and message.get("value"):
            return str(message["value"]).replace("<image>", "").replace("<video>", "").strip()
    return None


def extract_text(row, text_column):
    if text_column == "conversations":
        text = parse_conversations(row[text_column])
        if text:
            return text
    value = row.get(text_column) if text_column else None
    if value is not None:
        return str(value)
    if "conversations" in row:
        text = parse_conversations(row["conversations"])
        if text:
            return text
    raise ValueError("Could not extract text/caption from row.")


def find_columns(dataset):
    video_columns = [
        name for name in dataset.column_names
        if name.lower() in ("video", "videos", "mp4", "file", "path")
    ]
    text_columns = [
        name for name in dataset.column_names
        if name.lower() in ("text", "caption", "description", "sentence", "conversations")
    ]
    if not video_columns:
        raise ValueError(
            "No video-like column found. "
            f"Columns: {dataset.column_names}. "
            "Tip: hf-internal-testing/tiny-video-dataset is stored as text files and is not "
            "a paired table when loaded with datasets. Try the default LLaVA-Video config "
            "or run without --hf to generate local synthetic videos."
        )
    if not text_columns:
        raise ValueError(f"No text-like column found. Columns: {dataset.column_names}")
    return video_columns[0], text_columns[0]


def prepare_hf_dataset(args):
    from datasets import load_dataset

    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    log("Step 1/4: loading dataset metadata from Hugging Face")
    config = args.config or None
    log(f"  dataset={args.dataset} config={config or '<none>'} split={args.split}")
    if config:
        dataset = load_dataset(args.dataset, config, split=args.split, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    log(f"  rows={len(dataset)} columns={dataset.column_names}")

    video_column, text_column = find_columns(dataset)
    log(f"Step 2/4: using video column '{video_column}' and text column '{text_column}'")

    manifest_rows = []
    archive_paths = resolve_llava_archives(args)
    if archive_paths:
        log("Using LLaVA archive(s):")
        for archive_path in archive_paths:
            log(f"  {archive_path}")
    scan_limit = min(max(args.scan_rows, args.max_rows), len(dataset))
    log(f"Step 3/4: materializing {args.max_rows} video files from the first {scan_limit} rows")
    for index in range(scan_limit):
        if len(manifest_rows) >= args.max_rows:
            break
        row = dataset[index]
        suffix = ".mp4"
        video_value = row[video_column]
        if isinstance(video_value, dict):
            source_name = video_value.get("path") or video_value.get("filename") or ""
            suffix = Path(source_name).suffix or suffix
        elif isinstance(video_value, str):
            suffix = Path(video_value).suffix or suffix

        output_path = videos_dir / f"sample_{len(manifest_rows):03d}{suffix}"
        ok, source = copy_or_write_video(
            video_value,
            args.dataset,
            output_path,
            args.cache_dir,
            config=config,
            archive_paths=archive_paths,
        )
        if not ok:
            log(f"  row {index}: skipped ({source})")
            continue

        capture = cv2.VideoCapture(str(output_path))
        decoded = capture.isOpened()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if decoded else 0
        capture.release()
        if not decoded or frame_count <= 0:
            log(f"  row {index}: skipped (OpenCV cannot decode {output_path})")
            output_path.unlink(missing_ok=True)
            continue

        rel_path = output_path.relative_to(output_dir)
        text = extract_text(row, text_column)
        manifest_rows.append({
            "video_path": str(rel_path),
            "text": text,
        })
        log(f"  row {index}: saved {rel_path} ({frame_count} frames, source={source})")

    if not manifest_rows:
        raise RuntimeError("No videos were materialized. Try increasing --scan-rows or check dataset access.")

    manifest_path = output_dir / "manifest.csv"
    log("Step 4/4: writing manifest")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    log(f"Done. Manifest: {manifest_path}")
    log(f"Train with: python .\\model\\tiny_video_text_train.py --manifest \"{manifest_path}\" --epochs 20 --batch-size 2")


def write_synthetic_video(path, color, shape, direction):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")

    for frame_index in range(24):
        frame = np.full((96, 96, 3), 245, dtype=np.uint8)
        progress = frame_index / 23.0
        if direction == "right":
            x, y = int(8 + progress * 56), 34
        elif direction == "left":
            x, y = int(64 - progress * 56), 34
        elif direction == "down":
            x, y = 36, int(8 + progress * 56)
        else:
            x, y = 36, int(64 - progress * 56)

        if shape == "square":
            cv2.rectangle(frame, (x, y), (x + 22, y + 22), color, -1)
        else:
            cv2.circle(frame, (x + 12, y + 12), 13, color, -1)
        writer.write(frame)
    writer.release()


def prepare_synthetic(args):
    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    log("Creating local synthetic video-text dataset")
    samples = [
        ("videos/red_square_right.mp4", "a red square moves to the right across a pale background", (0, 0, 255), "square", "right"),
        ("videos/blue_circle_left.mp4", "a blue circle moves to the left across a pale background", (255, 0, 0), "circle", "left"),
        ("videos/green_square_down.mp4", "a green square moves downward across a pale background", (0, 180, 0), "square", "down"),
        ("videos/yellow_circle_up.mp4", "a yellow circle moves upward across a pale background", (0, 210, 210), "circle", "up"),
    ]
    manifest_rows = []
    for rel_path, text, color, shape, direction in samples:
        path = output_dir / rel_path
        write_synthetic_video(path, color, shape, direction)
        manifest_rows.append({"video_path": rel_path, "text": text})
        log(f"  saved {rel_path}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    log(f"Done. Manifest: {manifest_path}")
    log(f"Train with: python .\\model\\tiny_video_text_train.py --manifest \"{manifest_path}\" --epochs 20 --batch-size 2")


def main():
    args = parser.parse_args()
    if args.synthetic or not args.hf:
        prepare_synthetic(args)
    else:
        prepare_hf_dataset(args)


if __name__ == "__main__":
    main()
