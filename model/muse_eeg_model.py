"""
Object recognition Things-EEG2 dataset
use 250 Hz data

The code is modified from https://github.com/eeyhsong/NICE-EEG

MUSE-Nerv-*: Use Enc_nervformer_eeg as EEG Encoder
MUSE-Nerv-GA: Need to modifify NervFormerEEGModel
"""
import hashlib
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
from einops import rearrange

from torch_geometric.nn import GATConv


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # self.instance_norm = nn.InstanceNorm2d(num_features=512)
        self.instance_norm = nn.InstanceNorm2d(num_features=1)


    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.instance_norm(x)

        x = self.tsconv(x)
        x = self.projection(x)
        return x



class STConvEEGModel(nn.Module):
    def __init__(self, output_dim):
        super(STConvEEGModel, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(num_features=1)

        self.gatnn = EEG_GAT()
        
        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (63, 1), (1, 1)),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):  
        x = self.stconv(x)
        #####MUSE-GA######
        # x = self.gatnn(x)
        ##################
        output_features = self.projection(x)

        return output_features
    

class NervFormerEEGModel(nn.Module):
    def __init__(self, output_dim):
        super(NervFormerEEGModel, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(num_features=1)

        self.gatnn = EEG_GAT()

        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (63, 1), (1, 1)),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.self_attn_ts = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.self_attn_st = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.cross_attn = nn.MultiheadAttention(embed_dim=40, num_heads=8, dropout=0.75)

        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2880, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_dim),
        )

        self.norm1 = nn.LayerNorm(40) #d_model=40
        self.norm2 = nn.LayerNorm(40)
        self.norm3 = nn.LayerNorm(40)


    def forward(self, x):
        x = self.instance_norm(x)
        ##### Nerv-GA#####
        # x = self.gatnn(x)
        ##################
        ts_features = self.tsconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        st_features = self.stconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        # Attention is applied over the 250 time steps. 
        bf_ts_features, _ = self.self_attn_ts(ts_features, ts_features, ts_features)
        bf_st_features, _ = self.self_attn_st(st_features, st_features, st_features)
        # LayerNorm
        bf_ts_features = self.norm1(bf_ts_features + ts_features)
        bf_st_features = self.norm2(bf_st_features + st_features)
        combined_features = torch.cat((bf_ts_features, bf_st_features), dim=0) # need to cat?
        cf_combined_features, _ = self.cross_attn(combined_features, combined_features, combined_features)
        final_combined_features = self.norm3(cf_combined_features + combined_features)
        final_combined_features = final_combined_features.permute(1, 0, 2).flatten(1)
        output_features = self.feed_forward(final_combined_features) #[1000, 1440]

        return output_features


class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        self.num_channels = 63
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        # Reshape x to (batch_size*num_channels, num_features) to pass through GATConv
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        # print("FlattenHead:: ", x.shape) #torch.Size([1000, 1440])
        return x


class Enc_nervformer_eeg(nn.Sequential):
    def __init__(self, output_dim = 1440):
        super().__init__(
            NervFormerEEGModel(output_dim),
            FlattenHead()
        )

class Enc_muse_eeg(nn.Sequential):
    def __init__(self, output_dim = 1440):
        super().__init__(
            STConvEEGModel(output_dim),
            FlattenHead()
        )


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=10, n_layers=4, input_dim=768, output_dim=768,
                 use_quantum=False, use_classical_replacement=False):
        super(QuantumLayer, self).__init__()
        self.use_quantum = use_quantum
        self.use_classical_replacement = use_classical_replacement

        if self.use_quantum:
            import pennylane as qml

            self.fc_in = nn.Linear(input_dim, n_qubits)
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface='torch')
            def quantum_circuit(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
            self.fc_out = nn.Linear(n_qubits, output_dim)
        elif self.use_classical_replacement:
            self.classical_layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )

    def forward(self, x):
        if self.use_quantum:
            x = torch.tanh(self.fc_in(x)) * np.pi
            x = self.qlayer(x)
            return self.fc_out(x)
        if self.use_classical_replacement:
            return self.classical_layer(x)
        return x
        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5, n_qubits=10, n_layers=4,
                 use_quantum=False, use_classical_replacement=False):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            QuantumLayer(n_qubits, n_layers, proj_dim, proj_dim, use_quantum, use_classical_replacement),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3, n_qubits=10, n_layers=4,
                 use_quantum=False, use_classical_replacement=False):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            QuantumLayer(n_qubits, n_layers, proj_dim, proj_dim, use_quantum, use_classical_replacement),
        ) 

    def forward(self, x):
        return x


def _l2_normalize(array, eps=1e-8):
    norm = np.linalg.norm(array)
    if norm < eps:
        return array
    return array / norm


def _pad_or_trim(array, dim):
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape[0] >= dim:
        return array[:dim]
    return np.pad(array, (0, dim - array.shape[0]))


class VideoTextFeatureExtractor:
    def __init__(self, feature_dim=768, num_frames=8, backend="clip",
                 clip_model_name="openai/clip-vit-base-patch32", device="cpu",
                 clip_batch_size=16, video_pooling="mean",
                 video_encoder_model_name="microsoft/xclip-base-patch32",
                 text_pooling="truncate", frame_sampling="uniform",
                 video_chunk_stride=None):
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.backend = "clip_frame" if backend == "clip" else backend
        self.clip_model_name = clip_model_name
        self.video_encoder_model_name = video_encoder_model_name
        self.device = torch.device(device)
        self.clip_batch_size = clip_batch_size
        self.video_pooling = video_pooling
        if text_pooling not in {"truncate", "chunk_mean", "chunks"}:
            raise ValueError(f"Unknown text_pooling: {text_pooling}")
        if frame_sampling not in {"uniform", "all"}:
            raise ValueError(f"Unknown frame_sampling: {frame_sampling}")
        self.text_pooling = text_pooling
        self.frame_sampling = frame_sampling
        self.video_chunk_stride = video_chunk_stride
        self.last_text_chunk_count = 1
        self.last_video_chunk_count = 1
        self.clip_model = None
        self.clip_processor = None
        self.video_encoder_model = None
        self.video_encoder_processor = None
        self.video_encoder_num_frames = None
        if self.backend == "clip_frame":
            self._load_clip()
        if self.backend == "video_encoder":
            self._load_video_encoder()

    def _load_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError(
                "Could not import transformers CLIP. This is often caused by a "
                "torch/torchvision version mismatch. For torch 2.8.x, install a "
                "matching torchvision, for example: "
                "python -m pip install --upgrade torchvision==0.23.0"
            ) from exc

        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

    def _load_video_encoder(self):
        try:
            from transformers import XCLIPModel, XCLIPProcessor
        except Exception as exc:
            raise RuntimeError(
                "Could not import transformers XCLIP. Install a recent transformers "
                "version, then use --feature_backend video_encoder."
            ) from exc

        self.video_encoder_model = XCLIPModel.from_pretrained(self.video_encoder_model_name).to(self.device)
        self.video_encoder_model.eval()
        self.video_encoder_processor = XCLIPProcessor.from_pretrained(self.video_encoder_model_name)
        self.video_encoder_num_frames = int(
            getattr(self.video_encoder_model.config.vision_config, "num_frames", 8)
        )
        if self.frame_sampling == "all":
            print(
                "Video encoder frame window:",
                self.video_encoder_num_frames,
                "| frame sampling: all",
                "| using chunked video encoding",
            )
        elif self.num_frames != self.video_encoder_num_frames:
            print(
                "Video encoder frame window:",
                self.video_encoder_num_frames,
                "| sampled frames:",
                self.num_frames,
                "| using chunked video encoding",
            )

    @staticmethod
    def _text_max_length(model, processor):
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        max_length = getattr(text_config, "max_position_embeddings", None)
        if max_length is None:
            max_length = getattr(processor.tokenizer, "model_max_length", 77)
        if max_length is None or max_length > 10000:
            max_length = 77
        return int(max_length)

    def _chunk_text_inputs(self, processor, text, max_length):
        tokenizer = processor.tokenizer
        token_ids = tokenizer(
            str(text),
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
        chunk_size = max(1, max_length - special_count)
        chunks = [
            token_ids[start:start + chunk_size]
            for start in range(0, len(token_ids), chunk_size)
        ] or [[]]

        encoded_chunks = []
        for chunk in chunks:
            input_ids = tokenizer.build_inputs_with_special_tokens(chunk)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            encoded_chunks.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            })
        return tokenizer.pad(encoded_chunks, padding=True, return_tensors="pt"), len(chunks)

    def _encode_text_features(self, text, processor, get_text_features, max_length,
                              allowed_keys=None):
        if self.text_pooling in {"chunk_mean", "chunks"}:
            inputs, chunk_count = self._chunk_text_inputs(processor, text, max_length)
            self.last_text_chunk_count = chunk_count
        else:
            inputs = processor(
                text=[str(text)],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            self.last_text_chunk_count = 1

        feature_batches = []
        num_chunks = inputs["input_ids"].shape[0]
        for start in range(0, num_chunks, self.clip_batch_size):
            batch = {
                key: value[start:start + self.clip_batch_size].to(self.device)
                for key, value in inputs.items()
                if allowed_keys is None or key in allowed_keys
            }
            with torch.no_grad():
                features = get_text_features(**batch)
            feature_batches.append(F.normalize(features, dim=1))

        features = torch.cat(feature_batches, dim=0)
        features = F.normalize(features, dim=1)
        if self.text_pooling == "chunks":
            return features.detach().cpu().numpy().astype(np.float32)

        features = features.mean(dim=0, keepdim=True)
        features = F.normalize(features, dim=1)
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def text_features(self, text):
        if self.backend == "clip_frame":
            return self._encode_text_features(
                text,
                self.clip_processor,
                self.clip_model.get_text_features,
                self._text_max_length(self.clip_model, self.clip_processor),
            )
        if self.backend == "video_encoder":
            return self._encode_text_features(
                text,
                self.video_encoder_processor,
                self.video_encoder_model.get_text_features,
                self._text_max_length(self.video_encoder_model, self.video_encoder_processor),
                allowed_keys={"input_ids", "attention_mask", "position_ids"},
            )

        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        features = np.zeros(self.feature_dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.feature_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            features[bucket] += sign
        return _l2_normalize(features)

    @staticmethod
    def _frame_stats(frame):
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1).astype(np.float32)
        hist = hist / max(hist.sum(), 1.0)
        edges = cv2.Canny(gray, 80, 160)
        return np.concatenate([
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            np.percentile(rgb, 10, axis=(0, 1)),
            np.percentile(rgb, 50, axis=(0, 1)),
            np.percentile(rgb, 90, axis=(0, 1)),
            hist,
            np.array([edges.mean() / 255.0], dtype=np.float32),
        ]).astype(np.float32)

    def _sample_video_frames(self, video_path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_sampling == "all":
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
            capture.release()
            if not frames:
                raise ValueError(f"OpenCV opened but decoded no frames: {video_path}")
            return frames

        if frame_count <= 0:
            frame_indices = list(range(self.num_frames))
        else:
            frame_indices = np.linspace(0, frame_count - 1, self.num_frames).astype(int).tolist()

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
        capture.release()

        if not frames:
            raise ValueError(f"OpenCV opened but decoded no frames: {video_path}")
        return frames

    def _clip_frame_features(self, video_path):
        frames = self._sample_video_frames(video_path)
        images = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames
        ]
        frame_features = []
        for start in range(0, len(images), self.clip_batch_size):
            batch_images = images[start:start + self.clip_batch_size]
            inputs = self.clip_processor(images=batch_images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            frame_features.append(features)
        features = torch.cat(frame_features, dim=0)
        features = F.normalize(features, dim=1)
        return features.detach().cpu()

    def _clip_video_features(self, video_path):
        frame_features = self._clip_frame_features(video_path)
        self.last_video_chunk_count = frame_features.shape[0]
        if self.video_pooling == "mean":
            features = frame_features.mean(dim=0, keepdim=True)
            features = F.normalize(features, dim=1)
            return features.squeeze(0).numpy().astype(np.float32)
        if self.video_pooling == "mean_std":
            mean = frame_features.mean(dim=0)
            std = frame_features.std(dim=0, unbiased=False)
            return torch.cat([mean, std], dim=0).numpy().astype(np.float32)
        if self.video_pooling in {"temporal_mlp", "attention", "temporal_attention", "chunks"}:
            return frame_features.numpy().astype(np.float32)
        raise ValueError(f"Unknown video_pooling: {self.video_pooling}")

    @staticmethod
    def _chunk_frames(frames, chunk_size, stride=None):
        stride = stride or chunk_size
        if stride <= 0:
            raise ValueError("video_chunk_stride must be positive")
        chunks = []
        for start in range(0, len(frames), stride):
            chunk = list(frames[start:start + chunk_size])
            if len(chunk) < chunk_size:
                chunk.extend([chunk[-1]] * (chunk_size - len(chunk)))
            chunks.append(chunk)
            if start + chunk_size >= len(frames):
                break
        return chunks

    def _xclip_chunk_features(self, rgb_frames):
        try:
            inputs = self.video_encoder_processor(videos=[rgb_frames], return_tensors="pt")
        except Exception:
            inputs = self.video_encoder_processor(videos=rgb_frames, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        with torch.no_grad():
            features = self.video_encoder_model.get_video_features(pixel_values=pixel_values)
        return F.normalize(features, dim=1)

    def _xclip_video_features(self, video_path):
        frames = self._sample_video_frames(video_path)
        rgb_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames
        ]
        chunk_size = self.video_encoder_num_frames or len(rgb_frames)
        stride = self.video_chunk_stride or chunk_size
        chunk_features = [
            self._xclip_chunk_features(chunk)
            for chunk in self._chunk_frames(rgb_frames, chunk_size, stride)
        ]
        features = torch.cat(chunk_features, dim=0)
        self.last_video_chunk_count = features.shape[0]
        if self.video_pooling == "chunks":
            return features.detach().cpu().numpy().astype(np.float32)
        features = features.mean(dim=0, keepdim=True)
        features = F.normalize(features, dim=1)
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def video_features(self, video_path):
        if self.backend == "clip_frame":
            return self._clip_video_features(video_path)
        if self.backend == "video_encoder":
            return self._xclip_video_features(video_path)

        frames = self._sample_video_frames(video_path)
        stats = []
        previous = None
        motion = []
        for frame in frames:
            stats.append(self._frame_stats(frame))
            small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            if previous is not None:
                motion.append(float(np.abs(small - previous).mean()))
            previous = small

        stats = np.stack(stats)
        motion_features = np.array([
            np.mean(motion) if motion else 0.0,
            np.std(motion) if motion else 0.0,
        ], dtype=np.float32)
        features = np.concatenate([stats.mean(axis=0), stats.std(axis=0), motion_features])
        self.last_video_chunk_count = 1
        return _l2_normalize(_pad_or_trim(features, self.feature_dim))

    @staticmethod
    def _stack_feature_arrays(feature_arrays, label):
        arrays = [np.asarray(array, dtype=np.float32) for array in feature_arrays]
        if all(array.ndim == 1 for array in arrays):
            return torch.tensor(np.stack(arrays), dtype=torch.float32)

        arrays = [
            array.reshape(1, -1) if array.ndim == 1 else array
            for array in arrays
        ]
        if not all(array.ndim == 2 for array in arrays):
            ranks = sorted({array.ndim for array in arrays})
            raise ValueError(f"{label} features must be 1D or 2D arrays, got ranks {ranks}")

        feature_dim = arrays[0].shape[1]
        if any(array.shape[1] != feature_dim for array in arrays):
            dims = sorted({array.shape[1] for array in arrays})
            raise ValueError(f"{label} feature dimensions do not match: {dims}")

        max_chunks = max(array.shape[0] for array in arrays)
        padded = np.zeros((len(arrays), max_chunks, feature_dim), dtype=np.float32)
        for index, array in enumerate(arrays):
            padded[index, :array.shape[0]] = array
        return torch.tensor(padded, dtype=torch.float32)

    def _text_features_from_dataframe(self, data):
        text_features = []
        chunk_counts = []
        for _, row in data.iterrows():
            text_features.append(self.text_features(row["text"]))
            chunk_counts.append(self.last_text_chunk_count)

        if chunk_counts:
            counts = np.asarray(chunk_counts, dtype=np.float32)
            print(
                "Text pooling:",
                self.text_pooling,
                "| chunks per caption min/median/mean/max:",
                int(counts.min()),
                float(np.median(counts)),
                float(counts.mean()),
                int(counts.max()),
            )
        return self._stack_feature_arrays(text_features, "text")

    def load_text_features(self, manifest_path):
        manifest_path = Path(manifest_path).resolve()
        data = pd.read_csv(manifest_path)
        if "text" not in data.columns:
            raise ValueError("Manifest is missing column: text")
        return self._text_features_from_dataframe(data)

    def load_manifest(self, manifest_path):
        manifest_path = Path(manifest_path).resolve()
        data = pd.read_csv(manifest_path)
        required_columns = {"video_path", "text"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        video_features = []
        video_chunk_counts = []
        decoded_paths = []
        for _, row in data.iterrows():
            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = manifest_path.parent / video_path
            video_features.append(self.video_features(video_path))
            video_chunk_counts.append(self.last_video_chunk_count)
            decoded_paths.append(str(video_path))
        text_features = self._text_features_from_dataframe(data)
        if video_chunk_counts:
            counts = np.asarray(video_chunk_counts, dtype=np.float32)
            print(
                "Video pooling:",
                self.video_pooling,
                "| frame sampling:",
                self.frame_sampling,
                "| chunks per video min/median/mean/max:",
                int(counts.min()),
                float(np.median(counts)),
                float(counts.mean()),
                int(counts.max()),
            )

        return (
            self._stack_feature_arrays(video_features, "video"),
            text_features,
            decoded_paths,
        )


class Proj_video(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3, n_qubits=10, n_layers=4,
                 use_quantum=True, projection_tail=None):
        if projection_tail is None:
            projection_tail = "quantum" if use_quantum else "identity"
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            ProjectionTail(proj_dim, n_qubits, n_layers, projection_tail),
        )


class ProjectionTail(nn.Module):
    def __init__(self, proj_dim=768, n_qubits=10, n_layers=4, projection_tail="identity"):
        super().__init__()
        self.projection_tail = projection_tail
        if projection_tail == "identity":
            self.tail = nn.Identity()
        elif projection_tail == "classical_bottleneck":
            self.tail = nn.Sequential(
                nn.Linear(proj_dim, n_qubits),
                nn.GELU(),
                nn.Linear(n_qubits, proj_dim),
            )
        elif projection_tail == "quantum":
            self.tail = QuantumLayer(n_qubits, n_layers, proj_dim, proj_dim, True, False)
        else:
            raise ValueError(
                "projection_tail must be one of: identity, quantum, classical_bottleneck"
            )

    def forward(self, x):
        return self.tail(x)
