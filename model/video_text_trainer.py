import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from contrastive_trainer_base import (
        ContrastiveTrainerBase,
        QuantumLayer,
        normalize_features,
        quantum_parameter_stats,
        random_retrieval_scores,
        retrieval_scores,
        set_global_seed,
        weights_init_normal,
    )
    from muse_eeg_model import VideoTextFeatureExtractor
except ImportError:
    from .contrastive_trainer_base import (
        ContrastiveTrainerBase,
        QuantumLayer,
        normalize_features,
        quantum_parameter_stats,
        random_retrieval_scores,
        retrieval_scores,
        set_global_seed,
        weights_init_normal,
    )
    from .muse_eeg_model import VideoTextFeatureExtractor


class PostEncoderQuantumAdapter(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=768,
        n_qubits=10,
        n_layers=4,
        residual_scale=0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.quantum = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, features):
        raw_features = align_last_feature_dim(features, self.output_dim)
        if features.dim() == 3:
            batch_size, num_chunks, input_dim = features.shape
            flat_features = features.reshape(batch_size * num_chunks, input_dim)
            quantum_delta = self.quantum(flat_features).reshape(batch_size, num_chunks, self.output_dim)
            active_mask = (features.norm(dim=-1, keepdim=True) > 1e-8).to(raw_features.dtype)
            return (raw_features + self.residual_scale * quantum_delta) * active_mask

        quantum_delta = self.quantum(features)
        return raw_features + self.residual_scale * quantum_delta


def split_indices(num_items, val_ratio, test_ratio, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator)
    if num_items < 4:
        return indices, indices[:0], indices

    test_count = max(1, int(round(num_items * test_ratio)))
    val_count = max(1, int(round(num_items * val_ratio)))
    if test_count + val_count >= num_items:
        test_count = max(1, num_items // 4)
        val_count = max(1, num_items // 4)

    test_idx = indices[:test_count]
    val_idx = indices[test_count:test_count + val_count]
    train_idx = indices[test_count + val_count:]
    if len(train_idx) < 2:
        train_idx = indices
        val_idx = indices[:0]
        test_idx = indices
    return train_idx, val_idx, test_idx


def manifest_file_hash(manifest_path):
    hasher = hashlib.sha256()
    with open(manifest_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pool_video_features(video_features):
    if video_features.dim() == 3:
        return video_features.mean(dim=1)
    return video_features


def align_last_feature_dim(features, output_dim):
    if features.shape[-1] == output_dim:
        return features
    if features.shape[-1] > output_dim:
        return features[..., :output_dim]
    pad_width = output_dim - features.shape[-1]
    padding = features.new_zeros(*features.shape[:-1], pad_width)
    return torch.cat([features, padding], dim=-1)


def align_feature_dim(features, output_dim):
    features = pool_video_features(features)
    if features.shape[1] == output_dim:
        return features
    if features.shape[1] > output_dim:
        return features[:, :output_dim]
    pad_width = output_dim - features.shape[1]
    padding = features.new_zeros(features.shape[0], pad_width)
    return torch.cat([features, padding], dim=1)


class VideoTextTrainer(ContrastiveTrainerBase):
    def __init__(self, args, adapter=None):
        if not args.manifest:
            raise ValueError("--manifest is required when --task video_text")
        self.adapter = adapter or getattr(args, "adapter", "quantum")
        if self.adapter not in {"raw", "quantum"}:
            raise ValueError("video_text adapter must be raw or quantum")
        self.projection_tail = self.adapter
        self.history = []
        self.best_val_scores = None
        self.checkpoint_path = None
        self.eval_scope = None
        super().__init__(args, "video_text", args.result_path, args.model_path)
        self.train_logit_scale = False
        self.loss_direction = "source_to_target"
        temperature = float(getattr(args, "temperature", 0.07))
        self.logit_scale.data.fill_(np.log(1.0 / temperature))

    def _build_extractor(self):
        return VideoTextFeatureExtractor(
            feature_dim=self.args.feature_dim,
            num_frames=self.args.num_frames,
            backend=self.args.feature_backend,
            clip_model_name=self.args.clip_model,
            device=self.device,
            clip_batch_size=self.args.clip_batch_size,
            video_pooling=self.args.video_pooling,
            video_encoder_model_name=self.args.video_encoder_model,
            text_pooling=getattr(self.args, "text_pooling", "truncate"),
            frame_sampling=getattr(self.args, "frame_sampling", "uniform"),
            video_chunk_stride=getattr(self.args, "video_chunk_stride", None),
        )

    def _load_or_extract_features(self, extractor=None):
        manifest_hash = manifest_file_hash(self.args.manifest)
        cache_path = self.args.feature_cache_path
        if cache_path is None:
            cache_path = os.path.join(self.result_path, "video_text_feature_cache.pt")
        metadata = {
            "manifest_hash": manifest_hash,
            "clip_model": self.args.clip_model,
            "video_encoder_model": self.args.video_encoder_model,
            "num_frames": self.args.num_frames,
            "feature_backend": "clip_frame" if self.args.feature_backend == "clip" else self.args.feature_backend,
            "video_pooling": self.args.video_pooling,
            "text_pooling": getattr(self.args, "text_pooling", "truncate"),
            "frame_sampling": getattr(self.args, "frame_sampling", "uniform"),
            "video_chunk_stride": getattr(self.args, "video_chunk_stride", None),
        }
        video_metadata_keys = [
            "manifest_hash",
            "clip_model",
            "video_encoder_model",
            "num_frames",
            "feature_backend",
            "video_pooling",
            "frame_sampling",
            "video_chunk_stride",
        ]
        if self.args.cache_features and os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu")
            if all(cache.get(key) == value for key, value in metadata.items()):
                print("Loaded feature cache:", os.path.abspath(cache_path))
                return cache["video_features"], cache["text_features"], cache["decoded_paths"]
            if all(cache.get(key) == metadata[key] for key in video_metadata_keys):
                print("Reusing cached video features and refreshing text features:", os.path.abspath(cache_path))
                if extractor is None:
                    extractor = self._build_extractor()
                text_features = extractor.load_text_features(self.args.manifest)
                torch.save(
                    {
                        "video_features": cache["video_features"].cpu(),
                        "text_features": text_features.cpu(),
                        "decoded_paths": cache["decoded_paths"],
                        **metadata,
                    },
                    cache_path,
                )
                return cache["video_features"], text_features, cache["decoded_paths"]
            print("Ignoring stale feature cache:", os.path.abspath(cache_path))

        if extractor is None:
            extractor = self._build_extractor()
        video_features, text_features, decoded_paths = extractor.load_manifest(self.args.manifest)
        if self.args.cache_features:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(
                {
                    "video_features": video_features.cpu(),
                    "text_features": text_features.cpu(),
                    "decoded_paths": decoded_paths,
                    **metadata,
                },
                cache_path,
            )
            print("Saved feature cache:", os.path.abspath(cache_path))
        return video_features, text_features, decoded_paths

    def prepare(self):
        set_global_seed(self.args.seed)
        print("Task: video_text")
        print("Using device:", self.device)
        print("Adapter:", self.adapter)
        print("Feature backend:", self.args.feature_backend)
        print("Video pooling:", self.args.video_pooling)
        print("Text pooling:", getattr(self.args, "text_pooling", "truncate"))
        print("Frame sampling:", getattr(self.args, "frame_sampling", "uniform"))
        print("Match pooling:", getattr(self.args, "match_pooling", "logmeanexp"))
        if self.args.feature_backend in {"clip", "clip_frame"}:
            print("CLIP model:", self.args.clip_model)
        if self.args.feature_backend == "video_encoder":
            print("Video encoder model:", self.args.video_encoder_model)
        print("Manifest:", os.path.abspath(self.args.manifest))

        self.video_features, self.text_features, decoded_paths = self._load_or_extract_features()
        print("Decoded video files:", len(decoded_paths))
        print("Feature shapes video/text:", tuple(self.video_features.shape), tuple(self.text_features.shape))
        if self.video_features.dim() not in {2, 3}:
            raise ValueError(
                f"video_text expects 2D or 3D video features, got {self.video_features.dim()}D"
            )

        video_input_dim = (
            self.video_features.shape[-1]
            if self.video_features.dim() == 3
            else self.video_features.shape[1]
        )
        if self.text_features.dim() not in {2, 3}:
            raise ValueError(
                f"video_text expects 2D or 3D text features, got {self.text_features.dim()}D"
            )

        self.video_chunks_per_item = self._sequence_mask(self._as_sequence(self.video_features)).sum(dim=1)
        self.text_chunks_per_item = self._sequence_mask(self._as_sequence(self.text_features)).sum(dim=1)
        print(
            "Cached chunks video/text median/max:",
            float(self.video_chunks_per_item.float().median().item()),
            int(self.video_chunks_per_item.max().item()),
            "/",
            float(self.text_chunks_per_item.float().median().item()),
            int(self.text_chunks_per_item.max().item()),
        )

        self.text_dim = self.text_features.shape[-1]
        self.num_items = self.video_features.shape[0]
        self.train_idx, self.val_idx, self.test_idx = split_indices(
            self.num_items,
            self.args.val_ratio,
            self.args.test_ratio,
            self.args.seed,
        )
        print("Split sizes train/val/test:", len(self.train_idx), len(self.val_idx), len(self.test_idx))
        torch.save(
            {
                "train_idx": self.train_idx,
                "val_idx": self.val_idx,
                "test_idx": self.test_idx,
                "seed": self.args.seed,
            },
            os.path.join(self.result_path, "video_text_split_indices.pt"),
        )

        batch_size = int(self.args.batch_size)
        if batch_size <= 0:
            batch_size = len(self.train_idx)
        train_dataset = torch.utils.data.TensorDataset(
            self.video_features[self.train_idx],
            self.text_features[self.train_idx],
        )
        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.video_model = None
        if self.adapter == "quantum":
            self.video_model = PostEncoderQuantumAdapter(
                input_dim=video_input_dim,
                output_dim=self.text_dim,
                n_qubits=self.args.n_qubits,
                n_layers=self.args.n_layers,
                residual_scale=getattr(self.args, "quantum_residual_scale", 0.1),
            ).to(self.device)
            self.video_model.apply(weights_init_normal)
            self.models = [self.video_model]
            self.trainable_modules = [self.video_model]
            checkpoint_name = f"video_text_quantum_adapter_{self.args.video_pooling}"
            self.checkpoint_path = os.path.join(self.model_path, checkpoint_name + ".pth")
        else:
            self.models = []
            self.trainable_modules = []
            self.checkpoint_path = None

    def train(self):
        if self.adapter == "raw":
            self.prepare()
            return self.after_training()
        return super().train()

    def train_loader(self):
        return self._train_loader

    def validation_batches(self):
        if len(self.val_idx) >= 2:
            yield self.video_features[self.val_idx], self.text_features[self.val_idx]
        else:
            yield self.video_features[self.train_idx], self.text_features[self.train_idx]

    def encode_video(self, video_batch, use_raw=False):
        if use_raw or self.adapter == "raw":
            return align_last_feature_dim(video_batch, self.text_dim)
        return self.video_model(video_batch)

    def encode_pair(self, video_batch, text_batch):
        return self.encode_video(video_batch), align_last_feature_dim(text_batch, self.text_dim)

    @staticmethod
    def _as_sequence(features):
        if features.dim() == 2:
            return features.unsqueeze(1)
        if features.dim() == 3:
            return features
        raise ValueError(f"Expected 2D or 3D features, got {features.dim()}D")

    @staticmethod
    def _sequence_mask(sequence):
        return sequence.norm(dim=-1) > 1e-8

    @staticmethod
    def _normalize_sequence(sequence):
        return sequence / sequence.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _pairwise_similarity(self, video_embed, text_embed):
        video_seq = self._normalize_sequence(self._as_sequence(video_embed))
        text_seq = self._normalize_sequence(self._as_sequence(text_embed))
        video_mask = self._sequence_mask(video_seq)
        text_mask = self._sequence_mask(text_seq)

        similarities = torch.einsum("bvd,ntd->bnvt", video_seq, text_seq)
        pair_mask = video_mask[:, None, :, None] & text_mask[None, :, None, :]
        valid_counts = pair_mask.sum(dim=(-1, -2)).clamp_min(1)
        pooling = getattr(self.args, "match_pooling", "logmeanexp")

        if pooling == "max":
            masked = similarities.masked_fill(~pair_mask, -1e9)
            return masked.amax(dim=(-1, -2))
        if pooling == "mean":
            masked = similarities.masked_fill(~pair_mask, 0.0)
            return masked.sum(dim=(-1, -2)) / valid_counts
        if pooling == "logmeanexp":
            temperature = float(getattr(self.args, "match_temperature", 0.07))
            scaled = similarities / max(temperature, 1e-6)
            masked = scaled.masked_fill(~pair_mask, -1e9)
            pooled = torch.logsumexp(masked.flatten(start_dim=2), dim=-1)
            return temperature * (pooled - valid_counts.float().log())
        raise ValueError(f"Unknown match_pooling: {pooling}")

    def _embedding_std_mean(self, video_embed):
        video_seq = self._as_sequence(video_embed)
        video_mask = self._sequence_mask(video_seq)
        active = video_seq[video_mask]
        if active.numel() == 0:
            return np.nan
        active = self._normalize_sequence(active)
        return active.std(dim=0, unbiased=False).mean().item()

    def compute_pair_losses(self, source_batch, target_batch, include_extra=True):
        labels = torch.arange(source_batch.shape[0], device=self.device)
        source_embed, target_embed = self.encode_pair(source_batch, target_batch)
        logits_per_source = self.logit_scale.exp() * self._pairwise_similarity(source_embed, target_embed)
        loss_source = self.criterion_cls(logits_per_source, labels)
        if self.loss_direction == "symmetric":
            logits_per_target = logits_per_source.t()
            loss_target = self.criterion_cls(logits_per_target, labels)
            contrastive_loss = (loss_source + loss_target) / 2
        else:
            loss_target = source_embed.new_tensor(0.0)
            contrastive_loss = loss_source
        extra_loss = (
            self.batch_extra_loss(source_batch, target_batch, source_embed, target_embed)
            if include_extra
            else source_embed.new_tensor(0.0)
        )
        structure_loss = source_embed.new_tensor(0.0)
        total_loss = contrastive_loss + extra_loss
        return {
            "total": total_loss,
            "contrastive": contrastive_loss,
            "source": loss_source,
            "target": loss_target,
            "structure": structure_loss,
            "extra": extra_loss,
        }

    def retrieval_loss(self, indices, use_raw=False):
        if len(indices) < 2:
            return np.nan
        self.set_eval_mode()
        with torch.no_grad():
            video_batch = self.video_features[indices].to(self.device)
            text_batch = self.text_features[indices].to(self.device)
            video_embed = self.encode_video(video_batch, use_raw=use_raw)
            text_embed = align_last_feature_dim(text_batch, self.text_dim)
            labels = torch.arange(len(indices), device=self.device)
            logits = self.logit_scale.exp() * self._pairwise_similarity(video_embed, text_embed)
            return self.criterion_cls(logits, labels).item()

    def save_best_checkpoint(self):
        if len(self.val_idx) >= 1:
            self.best_val_scores = self.evaluate_both_directions(self.val_idx)
        torch.save(
            {
                "adapter": self.adapter,
                "model": self.video_model.state_dict(),
                "logit_scale": self.logit_scale.detach().cpu(),
                "args": vars(self.args),
                "train_idx": self.train_idx,
                "val_idx": self.val_idx,
                "test_idx": self.test_idx,
            },
            self.checkpoint_path,
        )

    def evaluate_both_directions(self, indices, use_raw=False):
        self.set_eval_mode()
        if len(indices) < 1:
            nan_scores = {"top1": np.nan, "top3": np.nan, "top5": np.nan}
            return nan_scores, nan_scores
        with torch.no_grad():
            video_batch = self.video_features[indices].to(self.device)
            text_batch = self.text_features[indices].to(self.device)
            video_embed = self.encode_video(video_batch, use_raw=use_raw)
            text_embed = align_last_feature_dim(text_batch, self.text_dim)
            labels = torch.arange(len(indices), device=self.device)
            logits_v2t = 100.0 * self._pairwise_similarity(video_embed, text_embed)
            logits_t2v = logits_v2t.t()
            return retrieval_scores(logits_v2t, labels), retrieval_scores(logits_t2v, labels)

    def retrieval_diagnostics(self, indices, use_raw=False):
        self.set_eval_mode()
        if len(indices) < 1:
            return {
                "mean_rank": np.nan,
                "median_rank": np.nan,
                "positive_logit_mean": np.nan,
                "top_negative_logit_mean": np.nan,
                "margin_mean": np.nan,
                "margin_min": np.nan,
                "logit_std": np.nan,
                "embedding_std_mean": np.nan,
            }
        with torch.no_grad():
            video_batch = self.video_features[indices].to(self.device)
            text_batch = self.text_features[indices].to(self.device)
            video_embed = self.encode_video(video_batch, use_raw=use_raw)
            text_embed = align_last_feature_dim(text_batch, self.text_dim)
            logits = 100.0 * self._pairwise_similarity(video_embed, text_embed)
            positive_logits = logits.diag()
            if len(indices) > 1:
                negative_mask = torch.eye(len(indices), dtype=torch.bool, device=self.device)
                top_negative_logits = logits.masked_fill(negative_mask, -1e9).max(dim=1).values
                margins = positive_logits - top_negative_logits
                top_negative_logit_mean = top_negative_logits.mean().item()
                margin_mean = margins.mean().item()
                margin_min = margins.min().item()
            else:
                top_negative_logit_mean = np.nan
                margin_mean = np.nan
                margin_min = np.nan
            ranks = (logits > positive_logits[:, None]).sum(dim=1).float() + 1.0
            return {
                "mean_rank": ranks.mean().item(),
                "median_rank": ranks.median().item(),
                "positive_logit_mean": positive_logits.mean().item(),
                "top_negative_logit_mean": top_negative_logit_mean,
                "margin_mean": margin_mean,
                "margin_min": margin_min,
                "logit_std": logits.std(unbiased=False).item(),
                "embedding_std_mean": self._embedding_std_mean(video_embed),
            }

    def _diagnostic_fields(self, prefix, indices, use_raw=False):
        diagnostics = self.retrieval_diagnostics(indices, use_raw=use_raw)
        return {
            f"{prefix}_video_to_text_mean_rank": diagnostics["mean_rank"],
            f"{prefix}_video_to_text_median_rank": diagnostics["median_rank"],
            f"{prefix}_positive_logit_mean": diagnostics["positive_logit_mean"],
            f"{prefix}_top_negative_logit_mean": diagnostics["top_negative_logit_mean"],
            f"{prefix}_margin_mean": diagnostics["margin_mean"],
            f"{prefix}_margin_min": diagnostics["margin_min"],
            f"{prefix}_logit_std": diagnostics["logit_std"],
            f"{prefix}_video_embedding_std_mean": diagnostics["embedding_std_mean"],
        }

    def _metric_row(self, epoch, train_metrics, val_metrics, max_quantum_grad_norm):
        eval_val_idx = self.val_idx if len(self.val_idx) >= 1 else self.train_idx
        scores_v2t, scores_t2v = self.evaluate_both_directions(eval_val_idx)
        raw_v2t, raw_t2v = self.evaluate_both_directions(eval_val_idx, use_raw=True)
        random_scores = random_retrieval_scores(len(eval_val_idx))
        quantum_param_count = (
            quantum_parameter_stats(self.video_model)[0]
            if self.video_model is not None
            else 0
        )
        residual_scale = (
            self.video_model.residual_scale.detach().item()
            if self.video_model is not None
            else np.nan
        )
        row = {
            "epoch": epoch,
            "adapter": self.adapter,
            "selection_metric": "val_retrieval_loss",
            "train_retrieval_loss": train_metrics["contrastive"],
            "train_total_loss": train_metrics["total"],
            "val_retrieval_loss": val_metrics["contrastive"],
            "val_total_loss": val_metrics["total"],
            "val_video_to_text_top1": scores_v2t["top1"],
            "val_video_to_text_top3": scores_v2t["top3"],
            "val_video_to_text_top5": scores_v2t["top5"],
            "val_text_to_video_top1": scores_t2v["top1"],
            "val_text_to_video_top3": scores_t2v["top3"],
            "val_text_to_video_top5": scores_t2v["top5"],
            "raw_val_video_to_text_top1": raw_v2t["top1"],
            "raw_val_video_to_text_top3": raw_v2t["top3"],
            "raw_val_video_to_text_top5": raw_v2t["top5"],
            "raw_val_text_to_video_top1": raw_t2v["top1"],
            "raw_val_text_to_video_top3": raw_t2v["top3"],
            "raw_val_text_to_video_top5": raw_t2v["top5"],
            "random_top1": random_scores["top1"],
            "random_top3": random_scores["top3"],
            "random_top5": random_scores["top5"],
            "logit_scale": self.logit_scale.exp().item(),
            "temperature": 1.0 / self.logit_scale.exp().item(),
            "quantum_active": self.adapter == "quantum",
            "quantum_param_count": quantum_param_count,
            "quantum_grad_norm": max_quantum_grad_norm if self.adapter == "quantum" else np.nan,
            "quantum_residual_scale": residual_scale,
            "num_total": self.num_items,
            "num_train": len(self.train_idx),
            "num_val": len(self.val_idx),
            "num_test": len(self.test_idx),
            "num_gallery": len(eval_val_idx),
            "seed": self.args.seed,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "best_epoch": self.best_epoch,
            "video_pooling": self.args.video_pooling,
            "num_frames": self.args.num_frames,
            "feature_backend": "clip_frame" if self.args.feature_backend == "clip" else self.args.feature_backend,
            "text_pooling": getattr(self.args, "text_pooling", "truncate"),
            "frame_sampling": getattr(self.args, "frame_sampling", "uniform"),
            "video_chunk_stride": getattr(self.args, "video_chunk_stride", None),
            "match_pooling": getattr(self.args, "match_pooling", "logmeanexp"),
            "match_temperature": getattr(self.args, "match_temperature", np.nan),
            "video_chunks_median": float(self.video_chunks_per_item.float().median().item()),
            "video_chunks_max": int(self.video_chunks_per_item.max().item()),
            "text_chunks_median": float(self.text_chunks_per_item.float().median().item()),
            "text_chunks_max": int(self.text_chunks_per_item.max().item()),
            "clip_model": self.args.clip_model,
            "video_encoder_model": self.args.video_encoder_model,
        }
        row.update(self._diagnostic_fields("train", self.train_idx))
        row.update(self._diagnostic_fields("raw_train", self.train_idx, use_raw=True))
        row.update(self._diagnostic_fields("val", eval_val_idx))
        row.update(self._diagnostic_fields("raw_val", eval_val_idx, use_raw=True))
        return row

    def on_epoch_end(self, epoch, train_metrics, val_metrics, max_quantum_grad_norm):
        row = self._metric_row(epoch, train_metrics, val_metrics, max_quantum_grad_norm)
        self.history.append(row)
        print(
            "Epoch %03d train_retrieval=%.4f val_retrieval=%.4f val_v2t_top1=%.4f train_rank=%.2f val_rank=%.2f val_margin=%.3f scale=%.3f"
            % (
                row["epoch"],
                row["train_retrieval_loss"],
                row["val_retrieval_loss"],
                row["val_video_to_text_top1"],
                row["train_video_to_text_mean_rank"],
                row["val_video_to_text_mean_rank"],
                row["val_margin_mean"],
                row["logit_scale"],
            )
        )

    def _final_summary_row(self, eval_idx):
        scores_v2t, scores_t2v = self.evaluate_both_directions(eval_idx)
        raw_scores_v2t, raw_scores_t2v = self.evaluate_both_directions(eval_idx, use_raw=True)
        random_scores = random_retrieval_scores(len(eval_idx))
        val_idx = self.val_idx if len(self.val_idx) >= 1 else self.train_idx
        val_scores = self.best_val_scores
        if val_scores is None:
            val_scores = self.evaluate_both_directions(val_idx)
        val_loss = self.best_val_loss
        if not np.isfinite(val_loss):
            val_loss = self.retrieval_loss(val_idx, use_raw=(self.adapter == "raw"))
        quantum_param_count = (
            quantum_parameter_stats(self.video_model)[0]
            if self.video_model is not None
            else 0
        )
        residual_scale = (
            self.video_model.residual_scale.detach().item()
            if self.video_model is not None
            else np.nan
        )
        row = {
            "epoch": "best_test" if self.eval_scope == "test_split" else "best_all",
            "adapter": self.adapter,
            "selection_metric": "val_retrieval_loss",
            "train_retrieval_loss": np.nan,
            "train_total_loss": np.nan,
            "val_retrieval_loss": val_loss,
            "val_total_loss": val_loss,
            "val_video_to_text_top1": val_scores[0]["top1"],
            "val_video_to_text_top3": val_scores[0]["top3"],
            "val_video_to_text_top5": val_scores[0]["top5"],
            "val_text_to_video_top1": val_scores[1]["top1"],
            "val_text_to_video_top3": val_scores[1]["top3"],
            "val_text_to_video_top5": val_scores[1]["top5"],
            "test_video_to_text_top1": scores_v2t["top1"],
            "test_video_to_text_top3": scores_v2t["top3"],
            "test_video_to_text_top5": scores_v2t["top5"],
            "test_text_to_video_top1": scores_t2v["top1"],
            "test_text_to_video_top3": scores_t2v["top3"],
            "test_text_to_video_top5": scores_t2v["top5"],
            "raw_test_video_to_text_top1": raw_scores_v2t["top1"],
            "raw_test_video_to_text_top3": raw_scores_v2t["top3"],
            "raw_test_video_to_text_top5": raw_scores_v2t["top5"],
            "raw_test_text_to_video_top1": raw_scores_t2v["top1"],
            "raw_test_text_to_video_top3": raw_scores_t2v["top3"],
            "raw_test_text_to_video_top5": raw_scores_t2v["top5"],
            "random_top1": random_scores["top1"],
            "random_top3": random_scores["top3"],
            "random_top5": random_scores["top5"],
            "logit_scale": self.logit_scale.exp().item(),
            "temperature": 1.0 / self.logit_scale.exp().item(),
            "quantum_active": self.adapter == "quantum",
            "quantum_param_count": quantum_param_count,
            "quantum_grad_norm": np.nan,
            "quantum_residual_scale": residual_scale,
            "num_total": self.num_items,
            "num_train": len(self.train_idx),
            "num_val": len(self.val_idx),
            "num_test": len(self.test_idx),
            "num_gallery": len(eval_idx),
            "seed": self.args.seed,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "best_epoch": self.best_epoch,
            "eval_scope": self.eval_scope,
            "eval_on_all": self.args.eval_on_all,
            "video_pooling": self.args.video_pooling,
            "num_frames": self.args.num_frames,
            "feature_backend": "clip_frame" if self.args.feature_backend == "clip" else self.args.feature_backend,
            "text_pooling": getattr(self.args, "text_pooling", "truncate"),
            "frame_sampling": getattr(self.args, "frame_sampling", "uniform"),
            "video_chunk_stride": getattr(self.args, "video_chunk_stride", None),
            "match_pooling": getattr(self.args, "match_pooling", "logmeanexp"),
            "match_temperature": getattr(self.args, "match_temperature", np.nan),
            "video_chunks_median": float(self.video_chunks_per_item.float().median().item()),
            "video_chunks_max": int(self.video_chunks_per_item.max().item()),
            "text_chunks_median": float(self.text_chunks_per_item.float().median().item()),
            "text_chunks_max": int(self.text_chunks_per_item.max().item()),
            "clip_model": self.args.clip_model,
            "video_encoder_model": self.args.video_encoder_model,
        }
        row.update(self._diagnostic_fields("train", self.train_idx))
        row.update(self._diagnostic_fields("raw_train", self.train_idx, use_raw=True))
        row.update(self._diagnostic_fields("val", val_idx))
        row.update(self._diagnostic_fields("raw_val", val_idx, use_raw=True))
        row.update(self._diagnostic_fields("test", eval_idx))
        row.update(self._diagnostic_fields("raw_test", eval_idx, use_raw=True))
        return row

    def after_training(self):
        if self.adapter == "quantum":
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.video_model.load_state_dict(checkpoint["model"])
            if "logit_scale" in checkpoint:
                self.logit_scale.data = checkpoint["logit_scale"].to(self.device)

        eval_idx = torch.arange(self.num_items) if self.args.eval_on_all else self.test_idx
        self.eval_scope = "all_manifest" if self.args.eval_on_all else "test_split"
        if self.adapter == "raw":
            self.best_epoch = 0
            val_idx = self.val_idx if len(self.val_idx) >= 1 else self.train_idx
            self.best_val_loss = self.retrieval_loss(val_idx, use_raw=True)
            self.best_val_scores = self.evaluate_both_directions(val_idx, use_raw=True)

        summary_row = self._final_summary_row(eval_idx)
        self.history.append(summary_row)

        result_csv = os.path.join(self.result_path, "video_text_results.csv")
        pd.DataFrame(self.history).to_csv(result_csv, index=False)
        print("The best epoch is:", self.best_epoch)
        print(
            "%s %s Top1/Top3/Top5: v2t %.6f/%.6f/%.6f, t2v %.6f/%.6f/%.6f"
            % (
                self.adapter,
                self.eval_scope,
                summary_row["test_video_to_text_top1"],
                summary_row["test_video_to_text_top3"],
                summary_row["test_video_to_text_top5"],
                summary_row["test_text_to_video_top1"],
                summary_row["test_text_to_video_top3"],
                summary_row["test_text_to_video_top5"],
            )
        )
        print("Saved results to:", result_csv)
        if self.checkpoint_path is not None:
            print("Saved checkpoint to:", self.checkpoint_path)
        return summary_row
