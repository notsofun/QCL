import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from contrastive_trainer_base import (
        ContrastiveTrainerBase,
        ProjectionHead,
        compute_structure_loss,
        load_model_state,
        model_state_dict,
        normalize_features,
        set_global_seed,
        weights_init_normal,
    )
    from muse_eeg_model import Enc_muse_eeg, Enc_nervformer_eeg
except ImportError:
    from .contrastive_trainer_base import (
        ContrastiveTrainerBase,
        ProjectionHead,
        compute_structure_loss,
        load_model_state,
        model_state_dict,
        normalize_features,
        set_global_seed,
        weights_init_normal,
    )
    from .muse_eeg_model import Enc_muse_eeg, Enc_nervformer_eeg


class LegacyImageProjection(ProjectionHead):
    def __init__(self, n_qubits=10, n_layers=4):
        super().__init__(
            embedding_dim=768,
            proj_dim=768,
            drop_proj=0.3,
            n_qubits=n_qubits,
            n_layers=n_layers,
            projection_tail="quantum",
        )

    def forward(self, x):
        return x


class EegImageTrainer(ContrastiveTrainerBase):
    def __init__(self, args, subject):
        self.subject = subject
        self.model_idx = "test0_" + str(subject) + "_"
        self.local_path = os.path.abspath(args.data_root) + os.sep
        self.eeg_data_path = os.path.join(
            self.local_path, "Data", "Things-EEG2", "Preprocessed_data_250Hz"
        ) + os.sep
        self.img_data_path = os.path.join(
            self.local_path,
            "Data",
            "Things-EEG2",
            "DNN_feature_maps",
            "pca_feature_maps",
            args.dnn,
            "pretrained-True",
        ) + os.sep
        self.test_center_path = os.path.join(
            self.local_path, "Data", "Things-EEG2", "Image_set"
        ) + os.sep
        super().__init__(args, "eeg_image", args.result_path, args.model_path)
        self.train_logit_scale = bool(getattr(args, "train_logit_scale", False))
        self.structure_loss_weight = 0.0
        self.eeg_projection_tail = getattr(args, "eeg_projection_tail", "quantum")
        self.image_projection_tail = getattr(args, "image_projection_tail", "legacy_identity")
        self.projection_tail = self.eeg_projection_tail
        self.batch_size_test = 400
        self.log_write = None

    def _maybe_parallel(self, model):
        model = model.to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model

    def _load_eeg_data(self):
        print("self.nSub: ", self.subject)
        train_data = np.load(
            self.eeg_data_path
            + "sub-"
            + format(self.subject, "02")
            + "/preprocessed_eeg_training.npy",
            allow_pickle=True,
        )
        train_data = train_data["preprocessed_eeg_data"]
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(
            self.eeg_data_path
            + "sub-"
            + format(self.subject, "02")
            + "/preprocessed_eeg_test.npy",
            allow_pickle=True,
        )
        test_data = test_data["preprocessed_eeg_data"]
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.arange(200)
        return train_data, test_data, test_label

    def _load_image_data(self):
        train_img_feature = np.load(
            self.img_data_path + self.args.dnn + "_feature_maps_training.npy",
            allow_pickle=True,
        )
        test_img_feature = np.load(
            self.img_data_path + self.args.dnn + "_feature_maps_test.npy",
            allow_pickle=True,
        )
        return np.squeeze(train_img_feature), np.squeeze(test_img_feature)

    def prepare(self):
        set_global_seed(getattr(self.args, "active_seed", self.args.seed))
        self.log_write = open(
            os.path.join(self.result_path, "log_subject%d.txt" % self.subject),
            "w",
        )

        if getattr(self.args, "eeg_encoder", "muse") == "nervformer":
            self.eeg_encoder = self._maybe_parallel(Enc_nervformer_eeg())
        else:
            self.eeg_encoder = self._maybe_parallel(Enc_muse_eeg())
        self.eeg_projection = self._maybe_parallel(
            ProjectionHead(
                embedding_dim=1440,
                proj_dim=768,
                drop_proj=0.5,
                n_qubits=self.args.n_qubits,
                n_layers=self.args.n_layers,
                projection_tail=self.eeg_projection_tail,
            )
        )
        if self.image_projection_tail == "legacy_identity":
            self.image_projection = self._maybe_parallel(
                LegacyImageProjection(
                    n_qubits=self.args.n_qubits,
                    n_layers=self.args.n_layers,
                )
            )
        else:
            self.image_projection = self._maybe_parallel(
                ProjectionHead(
                    embedding_dim=768,
                    proj_dim=768,
                    drop_proj=0.3,
                    n_qubits=self.args.n_qubits,
                    n_layers=self.args.n_layers,
                    projection_tail=self.image_projection_tail,
                )
            )
        self.eeg_projection.apply(weights_init_normal)
        self.image_projection.apply(weights_init_normal)
        self.models = [self.eeg_encoder, self.eeg_projection, self.image_projection]

        train_eeg, test_eeg, test_label = self._load_eeg_data()
        train_img_feature, _ = self._load_image_data()
        self.test_center = torch.from_numpy(
            np.load(
                self.test_center_path + "center_" + self.args.dnn + ".npy",
                allow_pickle=True,
            )
        ).float()

        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740]).float()
        val_image = torch.from_numpy(train_img_feature[:740]).float()
        train_eeg = torch.from_numpy(train_eeg[740:]).float()
        train_image = torch.from_numpy(train_img_feature[740:]).float()
        print("train_eeg: ", train_eeg.shape)
        print("train_image: ", train_image.shape)

        train_dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self._train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self._val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_eeg).float(),
            torch.from_numpy(test_label).long(),
        )
        self._test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
        )
        print("initial define done.")

    def train_loader(self):
        return self._train_loader

    def validation_batches(self):
        return self._val_loader

    def encode_pair(self, eeg_batch, image_batch):
        eeg_features = self.eeg_encoder(eeg_batch)
        eeg_features = self.eeg_projection(eeg_features)
        image_features = self.image_projection(image_batch)
        return eeg_features, image_features

    def batch_extra_loss(self, eeg_batch, image_batch, source_embed, target_embed):
        weight = float(getattr(self.args, "eeg_raw_structure_loss_weight", 1.0))
        if weight <= 0:
            return source_embed.new_tensor(0.0)
        flattened_eeg = eeg_batch.view(eeg_batch.shape[0], -1)
        return weight * compute_structure_loss(flattened_eeg, image_batch)

    def save_best_checkpoint(self):
        torch.save(
            model_state_dict(self.eeg_encoder),
            os.path.join(self.model_path, self.model_idx + "Enc_custom_eeg_cls.pth"),
        )
        torch.save(
            model_state_dict(self.eeg_projection),
            os.path.join(self.model_path, self.model_idx + "Proj_eeg_cls.pth"),
        )
        torch.save(
            model_state_dict(self.image_projection),
            os.path.join(self.model_path, self.model_idx + "Proj_img_cls.pth"),
        )

    def on_epoch_end(self, epoch, train_metrics, val_metrics, max_quantum_grad_norm):
        print(
            "Epoch:",
            epoch - 1,
            "  Cos eeg: %.4f" % train_metrics["source"],
            "  Cos img: %.4f" % train_metrics["target"],
            "  loss val: %.4f" % val_metrics["total"],
        )
        self.log_write.write(
            "Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n"
            % (epoch - 1, train_metrics["source"], train_metrics["target"], val_metrics["total"])
        )

    def _load_best_checkpoint(self):
        load_model_state(
            self.eeg_encoder,
            os.path.join(self.model_path, self.model_idx + "Enc_custom_eeg_cls.pth"),
            self.device,
        )
        load_model_state(
            self.eeg_projection,
            os.path.join(self.model_path, self.model_idx + "Proj_eeg_cls.pth"),
            self.device,
        )
        load_model_state(
            self.image_projection,
            os.path.join(self.model_path, self.model_idx + "Proj_img_cls.pth"),
            self.device,
        )

    def after_training(self):
        self._load_best_checkpoint()
        self.set_eval_mode()
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0
        all_center = self.test_center.to(self.device)

        with torch.no_grad():
            if self.image_projection_tail != "legacy_identity":
                all_center = normalize_features(self.image_projection(all_center))
            for eeg, label in self._test_loader:
                eeg = eeg.to(self.device)
                label = label.to(self.device)
                test_features = self.eeg_projection(self.eeg_encoder(eeg))
                test_features = normalize_features(test_features)
                similarity = (100.0 * test_features @ all_center.t()).softmax(dim=-1)
                _, indices = similarity.topk(5)
                label = label.view(-1, 1)
                total += label.size(0)
                top1 += (label == indices[:, :1]).sum().item()
                top3 += (label == indices[:, :3]).sum().item()
                top5 += (label == indices).sum().item()

        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)
        print("The test Top1-%.6f, Top3-%.6f, Top5-%.6f" % (top1_acc, top3_acc, top5_acc))
        self.log_write.write("The best epoch is: %d\n" % self.best_epoch)
        self.log_write.write(
            "The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n"
            % (top1_acc, top3_acc, top5_acc)
        )
        self.log_write.close()
        return top1_acc, top3_acc, top5_acc


def train_eeg_image_subjects(args):
    accuracies = []
    accuracies3 = []
    accuracies5 = []
    for index in range(args.num_sub):
        seed_n = np.random.randint(args.seed)
        args.active_seed = seed_n
        print("seed is " + str(seed_n))
        print("Subject %d" % (index + 1))
        trainer = EegImageTrainer(args, index + 1)
        acc, acc3, acc5 = trainer.train()
        print("THE BEST ACCURACY IS " + str(acc))
        accuracies.append(acc)
        accuracies3.append(acc3)
        accuracies5.append(acc5)

    accuracies.append(np.mean(accuracies))
    accuracies3.append(np.mean(accuracies3))
    accuracies5.append(np.mean(accuracies5))
    columns = np.arange(1, args.num_sub + 1).tolist()
    columns.append("ave")
    pd_all = pd.DataFrame(columns=columns, data=[accuracies, accuracies3, accuracies5])
    pd_all.to_csv(os.path.join(args.result_path, "result.csv"))
