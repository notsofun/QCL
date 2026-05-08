import argparse
import copy
import datetime
import os
import time

import pandas as pd

try:
    from eeg_image_trainer import train_eeg_image_subjects
    from video_text_trainer import VideoTextTrainer
except ImportError:
    from .eeg_image_trainer import train_eeg_image_subjects
    from .video_text_trainer import VideoTextTrainer


gpus = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RESULT_PATH = os.path.join(BASE_DIR, "results") + os.sep
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model") + os.sep


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if value.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Refactored Quantum Contrastive Learning trainer"
    )
    parser.add_argument(
        "--task",
        default="eeg_image",
        choices=["eeg_image", "video_text"],
        help="training task: Things-EEG2 EEG-image, or video-description contrastive learning",
    )
    parser.add_argument("--dnn", default="clip", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument(
        "--num_sub",
        default=1,
        type=int,
        help="number of subjects used in the EEG-image experiments",
    )
    parser.add_argument(
        "-batch_size",
        "--batch-size",
        default=1000,
        type=int,
        metavar="N",
        help="mini-batch size",
    )
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument(
        "--data_root",
        default=BASE_DIR,
        type=str,
        help="repository/data root containing Data/Things-EEG2",
    )
    parser.add_argument("--result_path", default=DEFAULT_RESULT_PATH, type=str)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, type=str)
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        choices=["auto", "cpu", "cuda"],
        help="device: auto, cpu, cuda",
    )

    parser.add_argument(
        "--eeg_encoder",
        default="muse",
        choices=["muse", "nervformer"],
        help="EEG encoder used by the EEG-image task",
    )
    parser.add_argument(
        "--eeg_projection_tail",
        default="quantum",
        choices=["identity", "quantum"],
        help="EEG projection tail for the EEG-image task",
    )
    parser.add_argument(
        "--eeg_raw_structure_loss_weight",
        default=0.0,
        type=float,
        help="optional weight for the original raw EEG/image similarity-structure regularizer",
    )
    parser.add_argument(
        "--train_logit_scale",
        default=False,
        type=str2bool,
        nargs="?",
        const=True,
        help="train logit_scale in the EEG-image task. Default preserves old behavior",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        type=str,
        help="video_text manifest CSV with video_path,text columns",
    )
    parser.add_argument("--feature_dim", default=768, type=int)
    parser.add_argument("--proj_dim", default=768, type=int)
    parser.add_argument(
        "--num_frames",
        default=8,
        type=int,
        help=(
            "total frames sampled per video. For XCLIP video_encoder, frames are "
            "encoded in the model's fixed temporal window and pooled across chunks"
        ),
    )
    parser.add_argument(
        "--feature_backend",
        default="clip_frame",
        choices=["clip", "clip_frame", "handcraft", "video_encoder"],
    )
    parser.add_argument(
        "--video_pooling",
        default="mean",
        choices=["mean", "mean_std", "temporal_mlp", "attention", "temporal_attention", "chunks"],
    )
    parser.add_argument(
        "--text_pooling",
        default="chunk_mean",
        choices=["truncate", "chunk_mean", "chunks"],
        help="text feature pooling: truncate keeps the model context limit, chunk_mean averages chunks, chunks keeps all caption chunks for pairwise matching",
    )
    parser.add_argument(
        "--text_preprocess",
        default="none",
        choices=["none", "first_sentence", "first_n_words"],
        help="optional text shortening before tokenization, useful for raw encoder sanity checks",
    )
    parser.add_argument("--text_first_n_words", default=40, type=int)
    parser.add_argument(
        "--frame_sampling",
        default="uniform",
        choices=["uniform", "all"],
        help="video frame sampling: uniform samples --num_frames, all decodes every frame",
    )
    parser.add_argument(
        "--video_chunk_stride",
        default=None,
        type=int,
        help="stride between XCLIP video chunks. Defaults to the model frame window, usually 8",
    )
    parser.add_argument(
        "--match_pooling",
        default="logmeanexp",
        choices=["logmeanexp", "max", "mean"],
        help="pairwise pooling over video chunks x text chunks when either side keeps chunks",
    )
    parser.add_argument("--match_temperature", default=0.07, type=float)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32", type=str)
    parser.add_argument("--video_encoder_model", default="microsoft/xclip-base-patch32", type=str)
    parser.add_argument("--clip_batch_size", default=16, type=int)
    parser.add_argument(
        "--adapter",
        default="compare",
        choices=["compare", "raw", "quantum"],
        help="video_text adapter: raw evaluates frozen features, quantum trains a post-encoder quantum adapter, compare runs both",
    )
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--quantum_residual_scale", default=0.1, type=float)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--cache_features", default=False, type=str2bool, nargs="?", const=True)
    parser.add_argument("--feature_cache_path", default=None, type=str)
    parser.add_argument("--seeds", default=None, nargs="+", type=int)
    parser.add_argument("--n_qubits", default=10, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--eval_on_all", default=False, type=str2bool, nargs="?", const=True)

    parser.add_argument("--use_quantum", default=True, type=str2bool, nargs="?", const=True, help=argparse.SUPPRESS)
    parser.add_argument("--projection_tail", default=None, choices=["identity", "quantum"], help=argparse.SUPPRESS)
    parser.add_argument("--eval_identity_baseline", default=True, type=str2bool, nargs="?", const=True, help=argparse.SUPPRESS)
    parser.add_argument("--structure_loss_weight", default=0.0, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--train_text_projection", default=False, type=str2bool, nargs="?", const=True, help=argparse.SUPPRESS)
    return parser


def _write_video_text_compare_outputs(summaries, result_path):
    summary_df = pd.DataFrame(summaries)
    compare_path = os.path.join(result_path, "raw_vs_quantum_results.csv")
    summary_df.to_csv(compare_path, index=False)
    print("Saved raw-vs-quantum comparison to:", compare_path)

    raw_rows = summary_df[summary_df["adapter"] == "raw"]
    quantum_rows = summary_df[summary_df["adapter"] == "quantum"]
    if raw_rows.empty or quantum_rows.empty:
        return

    raw = raw_rows.iloc[-1]
    quantum = quantum_rows.iloc[-1]
    delta = {
        "seed": quantum["seed"],
        "eval_scope": quantum.get("eval_scope", "test_split"),
        "num_gallery": quantum["num_gallery"],
    }
    for metric in [
        "test_video_to_text_top1",
        "test_video_to_text_top3",
        "test_video_to_text_top5",
        "test_video_to_text_mean_rank",
        "test_margin_mean",
        "test_margin_min",
        "video_chunks_median",
        "video_chunks_max",
        "text_chunks_median",
        "text_chunks_max",
        "test_text_to_video_top1",
        "test_text_to_video_top3",
        "test_text_to_video_top5",
        "test_video_embedding_std_mean",
    ]:
        if metric not in raw or metric not in quantum:
            continue
        delta["raw_" + metric] = raw[metric]
        delta["quantum_" + metric] = quantum[metric]
        delta["delta_" + metric] = quantum[metric] - raw[metric]
    delta_path = os.path.join(result_path, "raw_vs_quantum_delta.csv")
    pd.DataFrame([delta]).to_csv(delta_path, index=False)
    print("Saved raw-vs-quantum deltas to:", delta_path)


def _run_video_text_adapter(args, adapter, result_path, model_path, feature_cache_path=None):
    run_args = copy.deepcopy(args)
    run_args.adapter = adapter
    run_args.result_path = result_path
    run_args.model_path = model_path
    if feature_cache_path is not None:
        run_args.cache_features = True
        run_args.feature_cache_path = feature_cache_path
    os.makedirs(run_args.result_path, exist_ok=True)
    os.makedirs(run_args.model_path, exist_ok=True)
    return VideoTextTrainer(run_args).train()


def train_video_text(args):
    if args.seeds:
        base_result_path = args.result_path
        base_model_path = args.model_path
        summaries = []
        for seed in args.seeds:
            seed_args = copy.deepcopy(args)
            seed_args.seed = seed
            seed_args.seeds = None
            seed_args.result_path = os.path.join(base_result_path, f"seed_{seed}")
            seed_args.model_path = os.path.join(base_model_path, f"seed_{seed}")
            os.makedirs(seed_args.result_path, exist_ok=True)
            os.makedirs(seed_args.model_path, exist_ok=True)
            if seed_args.adapter == "compare":
                shared_cache = seed_args.feature_cache_path or os.path.join(
                    seed_args.result_path,
                    "video_text_feature_cache.pt",
                )
                for adapter in ["raw", "quantum"]:
                    summaries.append(
                        _run_video_text_adapter(
                            seed_args,
                            adapter,
                            os.path.join(seed_args.result_path, adapter),
                            os.path.join(seed_args.model_path, adapter),
                            shared_cache,
                        )
                    )
            else:
                summaries.append(VideoTextTrainer(seed_args).train())

        summary_df = pd.DataFrame(summaries)
        metric_columns = [
            "test_video_to_text_top1",
            "test_video_to_text_top3",
            "test_video_to_text_top5",
            "test_video_to_text_mean_rank",
            "test_margin_mean",
            "video_chunks_median",
            "video_chunks_max",
            "text_chunks_median",
            "text_chunks_max",
            "test_text_to_video_top1",
            "test_text_to_video_top3",
            "test_text_to_video_top5",
        ]
        aggregate = {}
        for column in metric_columns:
            if column in summary_df:
                aggregate[f"mean_{column}"] = summary_df[column].mean()
                aggregate[f"std_{column}"] = summary_df[column].std(ddof=0)
        for column in [
            "adapter",
            "video_pooling",
            "text_pooling",
            "text_preprocess",
            "text_first_n_words",
            "frame_sampling",
            "video_chunk_stride",
            "match_pooling",
            "match_temperature",
            "num_frames",
            "feature_backend",
            "clip_model",
            "video_encoder_model",
            "batch_size",
            "lr",
            "num_total",
            "num_train",
            "num_val",
            "num_test",
        ]:
            if column in summary_df:
                aggregate[column] = summary_df[column].iloc[0]
        aggregate["seeds"] = " ".join(str(seed) for seed in args.seeds)
        summary_path = os.path.join(base_result_path, "video_text_ablation_summary.csv")
        pd.DataFrame([aggregate]).to_csv(summary_path, index=False)
        print("Saved multi-seed summary to:", summary_path)
        if args.adapter == "compare":
            _write_video_text_compare_outputs(summaries, base_result_path)
        return

    if args.adapter == "compare":
        base_result_path = args.result_path
        base_model_path = args.model_path
        shared_cache = args.feature_cache_path or os.path.join(
            base_result_path,
            "video_text_feature_cache.pt",
        )
        summaries = []
        for adapter in ["raw", "quantum"]:
            summaries.append(
                _run_video_text_adapter(
                    args,
                    adapter,
                    os.path.join(base_result_path, adapter),
                    os.path.join(base_model_path, adapter),
                    shared_cache,
                )
            )
        _write_video_text_compare_outputs(summaries, base_result_path)
        return

    VideoTextTrainer(args).train()


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.result_path = os.path.abspath(args.result_path)
    args.model_path = os.path.abspath(args.model_path)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    if args.task == "video_text":
        train_video_text(args)
        return

    starttime = datetime.datetime.now()
    train_eeg_image_subjects(args)
    endtime = datetime.datetime.now()
    print("EEG-image duration: " + str(endtime - starttime))


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
