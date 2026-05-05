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
        choices=["identity", "quantum", "classical_bottleneck"],
        help="EEG projection tail for the EEG-image task",
    )
    parser.add_argument(
        "--image_projection_tail",
        default="legacy_identity",
        choices=["legacy_identity", "identity", "quantum", "classical_bottleneck"],
        help="image projection for EEG-image. legacy_identity preserves qcl_train.py behavior",
    )
    parser.add_argument(
        "--eeg_raw_structure_loss_weight",
        default=1.0,
        type=float,
        help="weight for the original raw EEG/image similarity-structure loss",
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
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument(
        "--feature_backend",
        default="clip_frame",
        choices=["clip", "clip_frame", "handcraft", "video_encoder"],
    )
    parser.add_argument(
        "--video_pooling",
        default="mean",
        choices=["mean", "mean_std", "temporal_mlp", "attention", "temporal_attention"],
    )
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32", type=str)
    parser.add_argument("--video_encoder_model", default="microsoft/xclip-base-patch32", type=str)
    parser.add_argument("--clip_batch_size", default=16, type=int)
    parser.add_argument("--use_quantum", default=True, type=str2bool, nargs="?", const=True)
    parser.add_argument(
        "--projection_tail",
        default=None,
        choices=["identity", "quantum", "classical_bottleneck"],
        help="video_text projection tail. Defaults to quantum/identity from --use_quantum",
    )
    parser.add_argument(
        "--eval_identity_baseline",
        default=True,
        type=str2bool,
        nargs="?",
        const=True,
    )
    parser.add_argument("--structure_loss_weight", default=0.0, type=float)
    parser.add_argument("--train_text_projection", default=False, type=str2bool, nargs="?", const=True)
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
    return parser


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
            summaries.append(VideoTextTrainer(seed_args).train())

        summary_df = pd.DataFrame(summaries)
        metric_columns = [
            "test_video_to_text_top1",
            "test_video_to_text_top3",
            "test_video_to_text_top5",
            "test_text_to_video_top1",
            "test_text_to_video_top3",
            "test_text_to_video_top5",
            "identity_v2t_top1",
            "identity_v2t_top3",
            "identity_v2t_top5",
            "identity_t2v_top1",
            "identity_t2v_top3",
            "identity_t2v_top5",
        ]
        aggregate = {}
        for column in metric_columns:
            if column in summary_df:
                aggregate[f"mean_{column}"] = summary_df[column].mean()
                aggregate[f"std_{column}"] = summary_df[column].std(ddof=0)
        for column in [
            "projection_tail",
            "video_pooling",
            "num_frames",
            "feature_backend",
            "clip_model",
            "video_encoder_model",
            "train_text_projection",
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
