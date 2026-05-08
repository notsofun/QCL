import math
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

try:
    import pennylane as qml
except ModuleNotFoundError:
    qml = None

try:
    from muse_eeg_model import ResidualAdd
except ImportError:
    from .muse_eeg_model import ResidualAdd


def weights_init_normal(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_name="auto"):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_projection_tail(args):
    if getattr(args, "projection_tail", None) is not None:
        return args.projection_tail
    adapter = getattr(args, "adapter", None)
    if adapter in {"raw", "quantum"}:
        return adapter
    return "quantum" if getattr(args, "use_quantum", True) else "identity"


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def model_state_dict(model):
    return unwrap_model(model).state_dict()


def load_model_state(model, path, device):
    unwrap_model(model).load_state_dict(torch.load(path, map_location=device), strict=False)


def normalize_features(features):
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)


def compute_structure_loss(source_features, target_features):
    source_norm = normalize_features(source_features)
    target_norm = normalize_features(target_features)
    source_similarity = torch.mm(source_norm, source_norm.t())
    target_similarity = torch.mm(target_norm, target_norm.t())
    cross_similarity = F.cosine_similarity(source_similarity, target_similarity)
    return 1 - cross_similarity.mean()


def retrieval_scores(logits, labels, topk=(1, 3, 5)):
    max_k = min(max(topk), logits.shape[1])
    _, indices = logits.topk(max_k, dim=1)
    scores = {}
    for k in topk:
        effective_k = min(k, logits.shape[1])
        scores[f"top{k}"] = (
            indices[:, :effective_k] == labels[:, None]
        ).any(dim=1).float().mean().item()
    return scores


def random_retrieval_scores(num_gallery, topk=(1, 3, 5)):
    if num_gallery <= 0:
        return {f"top{k}": np.nan for k in topk}
    return {f"top{k}": min(k, num_gallery) / num_gallery for k in topk}


def quantum_parameter_stats(model):
    param_count = 0
    grad_sq_sum = 0.0
    has_grad = False
    for name, param in model.named_parameters():
        if "qlayer" not in name.lower():
            continue
        param_count += param.numel()
        if param.grad is not None:
            grad_sq_sum += float(param.grad.detach().pow(2).sum().cpu())
            has_grad = True
    grad_norm = float(np.sqrt(grad_sq_sum)) if has_grad else 0.0
    return param_count, grad_norm


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=10, n_layers=4, input_dim=768, output_dim=768):
        super().__init__()
        if qml is None:
            raise ModuleNotFoundError("pennylane is required for projection_tail=quantum")

        self.fc_in = nn.Linear(input_dim, n_qubits)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc_out = nn.Linear(n_qubits, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc_in(x)) * math.pi
        x = self.qlayer(x)
        return self.fc_out(x)


class ProjectionTail(nn.Module):
    def __init__(self, proj_dim=768, n_qubits=10, n_layers=4, projection_tail="identity"):
        super().__init__()
        self.projection_tail = projection_tail
        if projection_tail in {"identity", "raw"}:
            self.tail = nn.Identity()
        elif projection_tail == "classical_bottleneck":
            self.tail = nn.Sequential(
                nn.Linear(proj_dim, n_qubits),
                nn.GELU(),
                nn.Linear(n_qubits, proj_dim),
            )
        elif projection_tail == "quantum":
            self.tail = QuantumLayer(n_qubits, n_layers, proj_dim, proj_dim)
        else:
            raise ValueError("projection_tail must be one of: raw, identity, quantum, classical_bottleneck")

    def forward(self, x):
        return self.tail(x)


class ProjectionHead(nn.Sequential):
    def __init__(
        self,
        embedding_dim=768,
        proj_dim=768,
        drop_proj=0.3,
        n_qubits=10,
        n_layers=4,
        projection_tail="identity",
    ):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
            ProjectionTail(proj_dim, n_qubits, n_layers, projection_tail),
        )


class IdentityProjection(nn.Module):
    def forward(self, x):
        return x


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def freeze_module(module):
    set_requires_grad(module, False)
    module.eval()


def unfreeze_module(module):
    set_requires_grad(module, True)
    module.train()


class ContrastiveTrainerBase(ABC):
    def __init__(self, args, task_name, result_path, model_path, device=None):
        self.args = args
        self.task_name = task_name
        self.result_path = os.path.abspath(result_path)
        self.model_path = os.path.abspath(model_path)
        self.device = device if device is not None else get_device(getattr(args, "device", "auto"))
        self.criterion_cls = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
        self.structure_loss_weight = float(getattr(args, "structure_loss_weight", 0.0))
        self.train_logit_scale = bool(getattr(args, "train_logit_scale", False))
        self.loss_direction = getattr(args, "loss_direction", "source_to_target")
        self.models = []
        self.trainable_modules = []
        self.frozen_modules = []
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def train_loader(self):
        raise NotImplementedError

    @abstractmethod
    def encode_pair(self, source_batch, target_batch):
        raise NotImplementedError

    @abstractmethod
    def validation_batches(self):
        raise NotImplementedError

    @abstractmethod
    def save_best_checkpoint(self):
        raise NotImplementedError

    @abstractmethod
    def after_training(self):
        raise NotImplementedError

    def trainable_parameters(self):
        params = []
        modules = self.trainable_modules if self.trainable_modules else self.models
        for model in modules:
            params.extend([param for param in model.parameters() if param.requires_grad])
        if self.train_logit_scale:
            params.append(self.logit_scale)
        return params

    def batch_extra_loss(self, source_batch, target_batch, source_embed, target_embed):
        return source_embed.new_tensor(0.0)

    def set_train_mode(self):
        for model in self.frozen_modules:
            model.eval()
        modules = self.trainable_modules if self.trainable_modules else self.models
        for model in modules:
            model.train()

    def set_eval_mode(self):
        for model in self.models:
            model.eval()
        for model in self.trainable_modules:
            model.eval()
        for model in self.frozen_modules:
            model.eval()

    def to_device_batch(self, batch):
        return [item.to(self.device) if torch.is_tensor(item) else item for item in batch]

    def compute_pair_losses(self, source_batch, target_batch, include_extra=True):
        labels = torch.arange(source_batch.shape[0], device=self.device)
        source_embed, target_embed = self.encode_pair(source_batch, target_batch)
        source_embed = normalize_features(source_embed)
        target_embed = normalize_features(target_embed)

        logits_per_source = self.logit_scale.exp() * source_embed @ target_embed.t()
        loss_source = self.criterion_cls(logits_per_source, labels)
        if self.loss_direction == "symmetric":
            logits_per_target = logits_per_source.t()
            loss_target = self.criterion_cls(logits_per_target, labels)
            contrastive_loss = (loss_source + loss_target) / 2
        else:
            loss_target = source_embed.new_tensor(0.0)
            contrastive_loss = loss_source
        if self.structure_loss_weight > 0:
            structure_loss = compute_structure_loss(source_embed, target_embed)
        else:
            structure_loss = source_embed.new_tensor(0.0)
        extra_loss = self.batch_extra_loss(source_batch, target_batch, source_embed, target_embed) if include_extra else source_embed.new_tensor(0.0)
        total_loss = contrastive_loss + self.structure_loss_weight * structure_loss + extra_loss
        return {
            "total": total_loss,
            "contrastive": contrastive_loss,
            "source": loss_source,
            "target": loss_target,
            "structure": structure_loss,
            "extra": extra_loss,
        }

    def validate(self):
        self.set_eval_mode()
        totals = {
            "total": 0.0,
            "contrastive": 0.0,
            "source": 0.0,
            "target": 0.0,
            "structure": 0.0,
            "extra": 0.0,
        }
        count = 0
        with torch.no_grad():
            for source_batch, target_batch in self.validation_batches():
                if source_batch.shape[0] < 2:
                    continue
                source_batch, target_batch = self.to_device_batch([source_batch, target_batch])
                losses = self.compute_pair_losses(source_batch, target_batch, include_extra=False)
                for key in totals:
                    totals[key] += losses[key].item()
                count += 1
        if count == 0:
            return {key: np.nan for key in totals}
        return {key: value / count for key, value in totals.items()}

    def on_epoch_end(self, epoch, train_metrics, val_metrics, max_quantum_grad_norm):
        print(
            "Epoch %03d train_total=%.4f val_total=%.4f scale=%.3f"
            % (epoch, train_metrics["total"], val_metrics["total"], self.logit_scale.exp().item())
        )

    def train(self):
        self.prepare()
        trainable_params = self.trainable_parameters()
        if not trainable_params:
            return self.after_training()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(getattr(self.args, "lr", 0.0002)),
            betas=(0.5, 0.999),
        )
        projection_tail = getattr(self, "projection_tail", None)
        quantum_modules = self.trainable_modules if self.trainable_modules else self.models
        quantum_param_count = sum(quantum_parameter_stats(model)[0] for model in quantum_modules)
        if projection_tail == "quantum" and quantum_param_count == 0:
            raise RuntimeError("projection_tail=quantum was requested, but no qlayer parameters were found.")

        for epoch in range(1, int(self.args.epoch) + 1):
            self.set_train_mode()
            totals = {
                "total": 0.0,
                "contrastive": 0.0,
                "source": 0.0,
                "target": 0.0,
                "structure": 0.0,
                "extra": 0.0,
            }
            batches = 0
            accum_counter = 0
            max_quantum_grad_norm = 0.0
            optimizer.zero_grad()

            for source_batch, target_batch in self.train_loader():
                if source_batch.shape[0] < 2:
                    continue
                source_batch, target_batch = self.to_device_batch([source_batch, target_batch])
                losses = self.compute_pair_losses(source_batch, target_batch, include_extra=True)
                (losses["total"] / self.grad_accum_steps).backward()
                for model in quantum_modules:
                    _, quantum_grad_norm = quantum_parameter_stats(model)
                    max_quantum_grad_norm = max(max_quantum_grad_norm, quantum_grad_norm)
                accum_counter += 1
                if accum_counter % self.grad_accum_steps == 0:
                    optimizer.step()
                    self.logit_scale.data.clamp_(max=np.log(100.0))
                    optimizer.zero_grad()

                for key in totals:
                    totals[key] += losses[key].item()
                batches += 1

            if accum_counter > 0 and accum_counter % self.grad_accum_steps != 0:
                optimizer.step()
                self.logit_scale.data.clamp_(max=np.log(100.0))
                optimizer.zero_grad()

            if projection_tail == "quantum" and batches > 0 and max_quantum_grad_norm <= 0.0:
                raise RuntimeError("projection_tail=quantum was requested, but quantum circuit parameters received zero gradient.")

            train_metrics = {key: value / max(batches, 1) for key, value in totals.items()}
            val_metrics = self.validate()
            if val_metrics["total"] <= self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.best_epoch = epoch
                self.save_best_checkpoint()
            self.on_epoch_end(epoch, train_metrics, val_metrics, max_quantum_grad_norm)

        return self.after_training()
