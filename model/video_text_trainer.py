import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from contrastive_trainer_base import (
        ContrastiveTrainerBase,
        ProjectionHead,
        normalize_features,
        random_retrieval_scores,
        resolve_projection_tail,
        retrieval_scores,
        set_global_seed,
        weights_init_normal,
    )
    from muse_eeg_model import VideoTextFeatureExtractor
except ImportError:
    from .contrastive_trainer_base import (
        ContrastiveTrainerBase,
        ProjectionHead,
        normalize_features,
        random_retrieval_scores,
        resolve_projection_tail,
        retrieval_scores,
        set_global_seed,
        weights_init_normal,
    )
    from .muse_eeg_model import VideoTextFeatureExtractor


class TextProjectionHead(ProjectionHead):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            embedding_dim=embedding_dim,
            proj_dim=proj_dim,
            drop_proj=drop_proj,
            projection_tail="identity",
        )


class TemporalVideoProjection(nn.Module):
    def __init__(
        self,
        frame_dim=768,
        proj_dim=768,
        video_pooling="temporal_mlp",
        drop_proj=0.3,
        n_qubits=10,
        n_layers=4,
        projection_tail="identity",
    ):
        super().__init__()
        self.video_pooling = video_pooling
        if video_pooling == "temporal_mlp":
            self.temporal = nn.GRU(frame_dim, frame_dim, batch_first=True)
            self.attention = None
        elif video_pooling in {"attention", "temporal_attention"}:
            self.temporal = None
            self.attention = nn.Sequential(
                nn.LayerNorm(frame_dim),
                nn.Linear(frame_dim, 1),
            )
        else:
            self.temporal = None
            self.attention = None
        self.projection = ProjectionHead(
            embedding_dim=frame_dim,
            proj_dim=proj_dim,
            drop_proj=drop_proj,
            n_qubits=n_qubits,
            n_layers=n_layers,
            projection_tail=projection_tail,
        )

    def forward(self, x):
        if x.dim() != 3:
            return self.projection(x)
        if self.video_pooling == "temporal_mlp":
            temporal_out, _ = self.temporal(x)
            pooled = temporal_out.mean(dim=1)
        elif self.video_pooling in {"attention", "temporal_attention"}:
            weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
            pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        else:
            pooled = x.mean(dim=1)
        return self.projection(pooled)


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


def identity_video_features(video_features, text_dim):
    if video_features.dim() == 3:
        return video_features.mean(dim=1)
    if video_features.shape[1] == text_dim:
        return video_features
    if video_features.shape[1] >= text_dim:
        return video_features[:, :text_dim]
    return None


class VideoTextTrainer(ContrastiveTrainerBase):
    def __init__(self, args):
        if not args.manifest:
            raise ValueError("--manifest is required when --task video_text")
        self.projection_tail = resolve_projection_tail(args)
        self.history = []
        self.best_val_scores = None
        self.checkpoint_path = None
        super().__init__(args, "video_text", args.result_path, args.model_path)

    def _load_or_extract_features(self, extractor):
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
        }
        if self.args.cache_features and os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu")
            if all(cache.get(key) == value for key, value in metadata.items()):
                print("Loaded feature cache:", os.path.abspath(cache_path))
                return cache["video_features"], cache["text_features"], cache["decoded_paths"]
            print("Ignoring stale feature cache:", os.path.abspath(cache_path))

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
        print("Projection tail:", self.projection_tail)
        print("Feature backend:", self.args.feature_backend)
        print("Video pooling:", self.args.video_pooling)
        print("Train text projection:", self.args.train_text_projection)
        if self.args.feature_backend in {"clip", "clip_frame"}:
            print("CLIP model:", self.args.clip_model)
        if self.args.feature_backend == "video_encoder":
            print("Video encoder model:", self.args.video_encoder_model)
        print("Manifest:", os.path.abspath(self.args.manifest))

        extractor = VideoTextFeatureExtractor(
            feature_dim=self.args.feature_dim,
            num_frames=self.args.num_frames,
            backend=self.args.feature_backend,
            clip_model_name=self.args.clip_model,
            device=self.device,
            clip_batch_size=self.args.clip_batch_size,
            video_pooling=self.args.video_pooling,
            video_encoder_model_name=self.args.video_encoder_model,
        )
        self.video_features, self.text_features, decoded_paths = self._load_or_extract_features(extractor)
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
        text_dim = self.text_features.shape[1]
        self.num_items = self.video_features.shape[0]
        self.train_idx, self.val_idx, self.test_idx = split_indices(
            self.num_items,
            self.args.val_ratio,
            self.args.test_ratio,
            self.args.seed,
        )
        print("Split sizes train/val/test:", len(self.train_idx), len(self.val_idx), len(self.test_idx))

        train_dataset = torch.utils.data.TensorDataset(
            self.video_features[self.train_idx],
            self.text_features[self.train_idx],
        )
        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        if self.video_features.dim() == 3:
            self.video_model = TemporalVideoProjection(
                frame_dim=video_input_dim,
                proj_dim=text_dim,
                video_pooling=self.args.video_pooling,
                n_qubits=self.args.n_qubits,
                n_layers=self.args.n_layers,
                projection_tail=self.projection_tail,
            ).to(self.device)
        else:
            self.video_model = ProjectionHead(
                embedding_dim=video_input_dim,
                proj_dim=text_dim,
                drop_proj=0.3,
                n_qubits=self.args.n_qubits,
                n_layers=self.args.n_layers,
                projection_tail=self.projection_tail,
            ).to(self.device)
        self.video_model.apply(weights_init_normal)

        self.text_model = None
        if self.args.train_text_projection:
            self.text_model = TextProjectionHead(embedding_dim=text_dim, proj_dim=text_dim).to(self.device)
            self.text_model.apply(weights_init_normal)

        self.models = [self.video_model]
        if self.text_model is not None:
            self.models.append(self.text_model)

        checkpoint_name = f"video_text_Proj_video_{self.projection_tail}_{self.args.video_pooling}"
        if self.args.train_text_projection:
            checkpoint_name += "_dual"
        self.checkpoint_path = os.path.join(self.model_path, checkpoint_name + ".pth")

    def train_loader(self):
        return self._train_loader

    def validation_batches(self):
        if len(self.val_idx) >= 2:
            yield self.video_features[self.val_idx], self.text_features[self.val_idx]
        else:
            yield self.video_features[self.train_idx], self.text_features[self.train_idx]

    def encode_pair(self, video_batch, text_batch):
        video_embed = self.video_model(video_batch)
        text_embed = self.text_model(text_batch) if self.text_model is not None else text_batch
        return video_embed, text_embed

    def save_best_checkpoint(self):
        if len(self.val_idx) >= 1:
            self.best_val_scores = self.evaluate_both_directions(self.val_idx)
        torch.save(
            {
                "model": self.video_model.state_dict(),
                "text_model": self.text_model.state_dict() if self.text_model is not None else None,
                "logit_scale": self.logit_scale.detach().cpu(),
                "args": vars(self.args),
                "projection_tail": self.projection_tail,
            },
            self.checkpoint_path,
        )

    def evaluate_direction(self, source_features, target_features, query_indices, gallery_indices):
        self.set_eval_mode()
        with torch.no_grad():
            query = source_features[query_indices].to(self.device)
            gallery = target_features[gallery_indices].to(self.device)
            query_embed, gallery_embed = self.encode_pair(query, gallery)
            query_embed = normalize_features(query_embed)
            gallery_embed = normalize_features(gallery_embed)
            logits = 100.0 * query_embed @ gallery_embed.t()
            labels = torch.arange(len(query_indices), device=self.device)
            return retrieval_scores(logits, labels)

    def evaluate_both_directions(self, indices):
        scores_v2t = self.evaluate_direction(
            self.video_features,
            self.text_features,
            indices,
            indices,
        )
        self.set_eval_mode()
        with torch.no_grad():
            text_query = self.text_features[indices].to(self.device)
            video_gallery = self.video_features[indices].to(self.device)
            video_gallery_embed = self.video_model(video_gallery)
            text_query_embed = self.text_model(text_query) if self.text_model is not None else text_query
            video_gallery_embed = normalize_features(video_gallery_embed)
            text_query_embed = normalize_features(text_query_embed)
            logits_t2v = 100.0 * text_query_embed @ video_gallery_embed.t()
            labels = torch.arange(len(indices), device=self.device)
            scores_t2v = retrieval_scores(logits_t2v, labels)
        return scores_v2t, scores_t2v

    def evaluate_identity(self, indices):
        video_identity = identity_video_features(self.video_features, self.text_features.shape[1])
        if video_identity is None or len(indices) < 1:
            nan_scores = {"top1": np.nan, "top3": np.nan, "top5": np.nan}
            return nan_scores, nan_scores
        with torch.no_grad():
            video_embed = normalize_features(video_identity[indices].to(self.device))
            text_embed = normalize_features(self.text_features[indices].to(self.device))
            labels = torch.arange(len(indices), device=self.device)
            logits_v2t = 100.0 * video_embed @ text_embed.t()
            logits_t2v = logits_v2t.t()
            return retrieval_scores(logits_v2t, labels), retrieval_scores(logits_t2v, labels)

    def on_epoch_end(self, epoch, train_metrics, val_metrics, max_quantum_grad_norm):
        eval_val_idx = self.val_idx if len(self.val_idx) >= 1 else self.train_idx
        scores_v2t, scores_t2v = self.evaluate_both_directions(eval_val_idx)
        random_scores = random_retrieval_scores(len(eval_val_idx))
        if self.args.eval_identity_baseline:
            identity_v2t, identity_t2v = self.evaluate_identity(eval_val_idx)
        else:
            identity_v2t = {"top1": np.nan, "top3": np.nan, "top5": np.nan}
            identity_t2v = {"top1": np.nan, "top3": np.nan, "top5": np.nan}
        scale_value = self.logit_scale.exp().item()
        row = {
            "epoch": epoch,
            "train_contrastive_loss": train_metrics["contrastive"],
            "train_structure_loss": train_metrics["structure"],
            "train_total_loss": train_metrics["total"],
            "val_contrastive_loss": val_metrics["contrastive"],
            "val_structure_loss": val_metrics["structure"],
            "val_total_loss": val_metrics["total"],
            "val_video_to_text_top1": scores_v2t["top1"],
            "val_video_to_text_top3": scores_v2t["top3"],
            "val_video_to_text_top5": scores_v2t["top5"],
            "val_text_to_video_top1": scores_t2v["top1"],
            "val_text_to_video_top3": scores_t2v["top3"],
            "val_text_to_video_top5": scores_t2v["top5"],
            "identity_video_to_text_top1": identity_v2t["top1"],
            "identity_video_to_text_top3": identity_v2t["top3"],
            "identity_video_to_text_top5": identity_v2t["top5"],
            "identity_text_to_video_top1": identity_t2v["top1"],
            "identity_text_to_video_top3": identity_t2v["top3"],
            "identity_text_to_video_top5": identity_t2v["top5"],
            "random_top1": random_scores["top1"],
            "random_top3": random_scores["top3"],
            "random_top5": random_scores["top5"],
            "logit_scale": scale_value,
            "temperature": 1.0 / scale_value,
            "quantum_active": self.projection_tail == "quantum",
            "quantum_param_count": sum(
                param.numel()
                for name, param in self.video_model.named_parameters()
                if "qlayer" in name.lower()
            ),
            "quantum_grad_norm": max_quantum_grad_norm if self.projection_tail == "quantum" else np.nan,
            "num_total": self.num_items,
            "num_train": len(self.train_idx),
            "num_val": len(self.val_idx),
            "num_test": len(self.test_idx),
            "seed": self.args.seed,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "best_epoch": self.best_epoch,
            "projection_tail": self.projection_tail,
            "use_quantum": self.args.use_quantum,
            "structure_loss_weight": self.args.structure_loss_weight,
            "grad_accum_steps": self.args.grad_accum_steps,
            "video_pooling": self.args.video_pooling,
            "num_frames": self.args.num_frames,
            "feature_backend": "clip_frame" if self.args.feature_backend == "clip" else self.args.feature_backend,
            "clip_model": self.args.clip_model,
            "video_encoder_model": self.args.video_encoder_model,
            "train_text_projection": self.args.train_text_projection,
        }
        self.history.append(row)
        print(
            "Epoch %03d train_total=%.4f val_total=%.4f val_v2t_top1=%.4f val_t2v_top1=%.4f scale=%.3f"
            % (
                row["epoch"],
                row["train_total_loss"],
                row["val_total_loss"],
                row["val_video_to_text_top1"],
                row["val_text_to_video_top1"],
                row["logit_scale"],
            )
        )

    def after_training(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.video_model.load_state_dict(checkpoint["model"])
        if self.text_model is not None and checkpoint.get("text_model") is not None:
            self.text_model.load_state_dict(checkpoint["text_model"])
        if "logit_scale" in checkpoint:
            self.logit_scale.data = checkpoint["logit_scale"].to(self.device)

        eval_idx = torch.arange(self.num_items) if self.args.eval_on_all else self.test_idx
        test_scores_v2t, test_scores_t2v = self.evaluate_both_directions(eval_idx)
        if self.args.eval_identity_baseline:
            identity_test_v2t, identity_test_t2v = self.evaluate_identity(eval_idx)
        else:
            identity_test_v2t = {"top1": np.nan, "top3": np.nan, "top5": np.nan}
            identity_test_t2v = {"top1": np.nan, "top3": np.nan, "top5": np.nan}
        random_scores = random_retrieval_scores(len(eval_idx))
        scale_value = self.logit_scale.exp().item()
        summary_row = {
            "epoch": "best_test",
            "train_contrastive_loss": np.nan,
            "train_structure_loss": np.nan,
            "train_total_loss": np.nan,
            "val_contrastive_loss": np.nan,
            "val_structure_loss": np.nan,
            "val_total_loss": self.best_val_loss,
            "val_video_to_text_top1": self.best_val_scores[0]["top1"] if self.best_val_scores else np.nan,
            "val_video_to_text_top3": self.best_val_scores[0]["top3"] if self.best_val_scores else np.nan,
            "val_video_to_text_top5": self.best_val_scores[0]["top5"] if self.best_val_scores else np.nan,
            "val_text_to_video_top1": self.best_val_scores[1]["top1"] if self.best_val_scores else np.nan,
            "val_text_to_video_top3": self.best_val_scores[1]["top3"] if self.best_val_scores else np.nan,
            "val_text_to_video_top5": self.best_val_scores[1]["top5"] if self.best_val_scores else np.nan,
            "test_video_to_text_top1": test_scores_v2t["top1"],
            "test_video_to_text_top3": test_scores_v2t["top3"],
            "test_video_to_text_top5": test_scores_v2t["top5"],
            "test_text_to_video_top1": test_scores_t2v["top1"],
            "test_text_to_video_top3": test_scores_t2v["top3"],
            "test_text_to_video_top5": test_scores_t2v["top5"],
            "identity_v2t_top1": identity_test_v2t["top1"],
            "identity_v2t_top3": identity_test_v2t["top3"],
            "identity_v2t_top5": identity_test_v2t["top5"],
            "identity_t2v_top1": identity_test_t2v["top1"],
            "identity_t2v_top3": identity_test_t2v["top3"],
            "identity_t2v_top5": identity_test_t2v["top5"],
            "identity_video_to_text_top1": identity_test_v2t["top1"],
            "identity_video_to_text_top3": identity_test_v2t["top3"],
            "identity_video_to_text_top5": identity_test_v2t["top5"],
            "identity_text_to_video_top1": identity_test_t2v["top1"],
            "identity_text_to_video_top3": identity_test_t2v["top3"],
            "identity_text_to_video_top5": identity_test_t2v["top5"],
            "random_top1": random_scores["top1"],
            "random_top3": random_scores["top3"],
            "random_top5": random_scores["top5"],
            "logit_scale": scale_value,
            "temperature": 1.0 / scale_value,
            "quantum_active": self.projection_tail == "quantum",
            "quantum_param_count": sum(
                param.numel()
                for name, param in self.video_model.named_parameters()
                if "qlayer" in name.lower()
            ),
            "quantum_grad_norm": np.nan,
            "num_total": self.num_items,
            "num_train": len(self.train_idx),
            "num_val": len(self.val_idx),
            "num_test": len(self.test_idx),
            "seed": self.args.seed,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "best_epoch": self.best_epoch,
            "projection_tail": self.projection_tail,
            "eval_on_all": self.args.eval_on_all,
            "use_quantum": self.args.use_quantum,
            "structure_loss_weight": self.args.structure_loss_weight,
            "grad_accum_steps": self.args.grad_accum_steps,
            "video_pooling": self.args.video_pooling,
            "num_frames": self.args.num_frames,
            "feature_backend": "clip_frame" if self.args.feature_backend == "clip" else self.args.feature_backend,
            "clip_model": self.args.clip_model,
            "video_encoder_model": self.args.video_encoder_model,
            "train_text_projection": self.args.train_text_projection,
        }
        self.history.append(summary_row)

        result_csv = os.path.join(self.result_path, "video_text_results.csv")
        pd.DataFrame(self.history).to_csv(result_csv, index=False)
        print("The best epoch is:", self.best_epoch)
        print(
            "Best checkpoint test Top1/Top3/Top5: "
            "v2t %.6f/%.6f/%.6f, t2v %.6f/%.6f/%.6f"
            % (
                test_scores_v2t["top1"],
                test_scores_v2t["top3"],
                test_scores_v2t["top5"],
                test_scores_t2v["top1"],
                test_scores_t2v["top3"],
                test_scores_t2v["top5"],
            )
        )
        print("Saved results to:", result_csv)
        print("Saved checkpoint to:", self.checkpoint_path)
        return summary_row
