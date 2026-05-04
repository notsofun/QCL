try:
    import pennylane as qml
except ModuleNotFoundError:
    qml = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
from einops import rearrange

from torch_geometric.nn import GATConv
import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

from muse_eeg_model import Enc_nervformer_eeg, Enc_muse_eeg, ResidualAdd, Proj_video, VideoTextFeatureExtractor


gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# result_path = '/home/NICE/results/' 
result_path = os.path.join(BASE_DIR, 'results') + os.sep
model_path = os.path.join(BASE_DIR, 'model') + os.sep

model_idx = 'test0'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--task', default='eeg_image', choices=['eeg_image', 'video_text'],
                    help='training task: original Things-EEG2 EEG-image, or video-description contrastive learning')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=1, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2024, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--data_root', default=BASE_DIR, type=str,
                    help='repository/data root containing Data/Things-EEG2')
parser.add_argument('--result_path', default=result_path, type=str,
                    help='directory for logs and CSV results')
parser.add_argument('--model_path', default=model_path, type=str,
                    help='directory for model checkpoints')
parser.add_argument('--manifest', default=None, type=str,
                    help='video_text task manifest CSV with video_path,text columns')
parser.add_argument('--feature_dim', default=768, type=int,
                    help='video_text fixed feature dimension')
parser.add_argument('--proj_dim', default=768, type=int,
                    help='video_text projection/contrastive dimension')
parser.add_argument('--num_frames', default=8, type=int,
                    help='number of video frames sampled for video_text features')
parser.add_argument('--feature_backend', default='clip', choices=['clip', 'handcraft'],
                    help='video_text feature extractor: CLIP frame/text embeddings or hand-crafted fallback')
parser.add_argument('--clip_model', default='openai/clip-vit-base-patch32', type=str,
                    help='Hugging Face CLIP model used when --feature_backend clip')
parser.add_argument('--clip_batch_size', default=16, type=int,
                    help='batch size for CLIP frame encoding')
parser.add_argument('--use_quantum', default=True, type=str2bool, nargs='?', const=True,
                    help='video_text: add the quantum layer to the video projection head')
parser.add_argument('--n_qubits', default=10, type=int,
                    help='video_text quantum qubit count')
parser.add_argument('--n_layers', default=4, type=int,
                    help='video_text quantum layer count')
parser.add_argument('--lr', default=0.0002, type=float,
                    help='video_text learning rate')
parser.add_argument('--val_ratio', default=0.1, type=float,
                    help='video_text validation split ratio')
parser.add_argument('--test_ratio', default=0.2, type=float,
                    help='video_text test split ratio')
parser.add_argument('--device', default='auto', type=str,
                    help='video_text device: auto, cpu, cuda')
parser.add_argument('--eval_on_all', default=False, type=str2bool, nargs='?', const=True,
                    help='video_text: evaluate the best checkpoint on all manifest rows instead of only the test split')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=10, n_layers=4, input_dim=768, output_dim=768):
        super(QuantumLayer, self).__init__()
        if qml is None:
            raise ModuleNotFoundError('pennylane is required for qcl_train.py')

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

    def forward(self, x):
        x = self.fc_in(x)
        x = self.qlayer(x)
        return self.fc_out(x)


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5, n_qubits=10, n_layers=4):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            QuantumLayer(n_qubits, n_layers, proj_dim, proj_dim),
        )

class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3, n_qubits=10, n_layers=4):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            QuantumLayer(n_qubits, n_layers, proj_dim, proj_dim),
        )
    def forward(self, x):
        return x 

class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch
        self.lambda_cen = 0.003
        self.alpha = 0.5
        self.proj_dim = 256
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.model_idx = 'test0_' + str(self.nSub) + '_'

        local_path = os.path.abspath(args.data_root) + os.sep
        self.result_path = os.path.abspath(args.result_path) + os.sep
        self.model_path = os.path.abspath(args.model_path) + os.sep
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        self.start_epoch = 0
        self.eeg_data_path = os.path.join(local_path, 'Data', 'Things-EEG2', 'Preprocessed_data_250Hz') + os.sep
        self.img_data_path = os.path.join(local_path, 'Data', 'Things-EEG2', 'DNN_feature_maps', 'pca_feature_maps', args.dnn, 'pretrained-True') + os.sep
        self.test_center_path = os.path.join(local_path, 'Data', 'Things-EEG2', 'Image_set') + os.sep

        self.log_write = open(self.result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.Enc_nervformer_eeg = Enc_nervformer_eeg().cuda()
        self.Enc_nervformer_eeg = nn.DataParallel(self.Enc_nervformer_eeg, device_ids=[i for i in range(len(gpus))])

        self.Enc_muse_eeg = Enc_muse_eeg().cuda()
        self.Enc_muse_eeg = nn.DataParallel(self.Enc_muse_eeg, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)
        
        print("self.nSub: ", self.nSub)
        train_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data']
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        # The training set includes 1654concepts × 10images × 4repetitions, 63 channels.
        # print("train_eeg1: ", train_eeg.shape) #  (16540, 1, 63, 250)
        train_img_feature, _ = self.get_image_data() 
        # Images were resized to 224×224 and normalized before being processed by the image encoder
        # print('train_img_feature1: ', train_img_feature.shape) # (16540, 768)
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        print('train_eeg: ', train_eeg.shape) # torch.Size([15800, 1, 63, 250])
        train_image = torch.from_numpy(train_img_feature[740:])
        print('train_image: ', train_image.shape) # torch.Size([15800, 768])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)

        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        # self.optimizer = torch.optim.AdamW(itertools.chain(self.Enc_nervformer_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer = torch.optim.AdamW(itertools.chain(self.Enc_muse_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for e in range(self.n_epochs):
            in_epoch = time.time()

            
            # self.Enc_nervformer_eeg.train()
            self.Enc_muse_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                # print("eeg: ", eeg.shape) # torch.Size([1000, 1, 63, 250])
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # print("img_features: ", img_features.shape) # torch.Size([1000, 768])
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # eeg cor
                flattened_eeg_data_tensor = eeg.view(eeg.shape[0], -1)  # [1000, 63*250]

                # calculate L2 norm
                eeg_data_norms = torch.norm(flattened_eeg_data_tensor, p=2, dim=1, keepdim=True)
                normalized_eeg_tensor = flattened_eeg_data_tensor / eeg_data_norms
                eeg_cos_similarity_matrix = torch.mm(normalized_eeg_tensor, normalized_eeg_tensor.transpose(0, 1)) # [1000, 1000]

                # image cor
                img_features_norms = torch.norm(img_features, p=2, dim=1, keepdim=True) # torch.Size([1000, 768])
                normalized_tensor = img_features / img_features_norms
                img_cos_similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.transpose(0, 1)) # [1000, 1000]
                eeg_img_cos_similarity = F.cosine_similarity(eeg_cos_similarity_matrix, img_cos_similarity_matrix)
                eeg_img_cos_sim_loss = 1 - eeg_img_cos_similarity.mean()


                # obtain the features
                # eeg_features = self.Enc_nervformer_eeg(eeg)
                eeg_features = self.Enc_muse_eeg(eeg)

                # print('eeg_features1: ', eeg_features.shape) # torch.Size([1000, 1440])

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                # print('eeg_features_proj: ', eeg_features.shape) #  torch.Size([1000, 768])
                img_features = self.Proj_img(img_features)
                # print("img_features_proj: ", img_features.shape) #  torch.Size([1000, 768])


                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss SK-InfoNCE loss
                loss = loss_cos + eeg_img_cos_sim_loss

                # INfoNCE loss
                # loss = loss_cos 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if (e + 1) % 1 == 0:
                # self.Enc_nervformer_eeg.eval()
                self.Enc_muse_eeg.eval()

                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        # veeg_features = self.Enc_nervformer_eeg(veeg)
                        veeg_features = self.Enc_muse_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1

                            # torch.save(self.Enc_nervformer_eeg.module.state_dict(), model_path + self.model_idx + 'Enc_custom_eeg_cls.pth')
                            torch.save(self.Enc_muse_eeg.module.state_dict(), self.model_path + self.model_idx + 'Enc_custom_eeg_cls.pth')

                            torch.save(self.Proj_eeg.module.state_dict(), self.model_path + self.model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), self.model_path + self.model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                    '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                    '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                    '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                    )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        # self.Enc_nervformer_eeg.load_state_dict(torch.load(model_path + self.model_idx + 'Enc_custom_eeg_cls.pth'), strict=False)
        self.Enc_muse_eeg.load_state_dict(torch.load(self.model_path + self.model_idx + 'Enc_custom_eeg_cls.pth'), strict=False)

        self.Proj_eeg.load_state_dict(torch.load(self.model_path + self.model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load(self.model_path + self.model_idx + 'Proj_img_cls.pth'), strict=False)

        # self.Enc_nervformer_eeg.eval()
        self.Enc_muse_eeg.eval()

        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))            

                # tfea = self.Proj_eeg(self.Enc_nervformer_eeg(teeg))
                tfea = self.Proj_eeg(self.Enc_muse_eeg(teeg))


                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        self.log_write.close()

        return top1_acc, top3_acc, top5_acc


def get_video_text_device(args):
    if args.device == 'cpu':
        return torch.device('cpu')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requested, but CUDA is not available')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def compute_structure_loss(source_features, target_features):
    source_norm = source_features / source_features.norm(dim=1, keepdim=True).clamp_min(1e-8)
    target_norm = target_features / target_features.norm(dim=1, keepdim=True).clamp_min(1e-8)
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
        scores[f'top{k}'] = (indices[:, :effective_k] == labels[:, None]).any(dim=1).float().mean().item()
    return scores


def evaluate_video_text(model, source_features, target_features, query_indices, gallery_indices, device):
    model.eval()
    with torch.no_grad():
        query = source_features[query_indices].to(device)
        gallery = target_features[gallery_indices].to(device)
        query_embed = model(query)
        query_embed = query_embed / query_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)
        gallery_embed = gallery / gallery.norm(dim=1, keepdim=True).clamp_min(1e-8)
        logits = 100.0 * query_embed @ gallery_embed.t()
        labels = torch.arange(len(query_indices), device=device)
        return retrieval_scores(logits, labels)


def evaluate_video_text_both_directions(model, video_features, text_features, indices, device):
    scores_v2t = evaluate_video_text(model, video_features, text_features, indices, indices, device)
    model.eval()
    with torch.no_grad():
        text_query = text_features[indices].to(device)
        video_gallery = video_features[indices].to(device)
        video_gallery_embed = model(video_gallery)
        video_gallery_embed = video_gallery_embed / video_gallery_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)
        text_query_embed = text_query / text_query.norm(dim=1, keepdim=True).clamp_min(1e-8)
        logits_t2v = 100.0 * text_query_embed @ video_gallery_embed.t()
        labels = torch.arange(len(indices), device=device)
        scores_t2v = retrieval_scores(logits_t2v, labels)
    return scores_v2t, scores_t2v


def train_video_text(args):
    if not args.manifest:
        raise ValueError('--manifest is required when --task video_text')

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = get_video_text_device(args)
    result_dir = os.path.abspath(args.result_path)
    model_dir = os.path.abspath(args.model_path)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print('Task: video_text')
    print('Using device:', device)
    print('Using quantum:', args.use_quantum)
    print('Feature backend:', args.feature_backend)
    if args.feature_backend == 'clip':
        print('CLIP model:', args.clip_model)
    print('Manifest:', os.path.abspath(args.manifest))

    extractor = VideoTextFeatureExtractor(
        feature_dim=args.feature_dim,
        num_frames=args.num_frames,
        backend=args.feature_backend,
        clip_model_name=args.clip_model,
        device=device,
        clip_batch_size=args.clip_batch_size,
    )
    video_features, text_features, decoded_paths = extractor.load_manifest(args.manifest)
    print('Decoded video files:', len(decoded_paths))
    print('Feature shapes video/text:', tuple(video_features.shape), tuple(text_features.shape))
    if video_features.shape[1] != text_features.shape[1]:
        raise ValueError(
            'video_text expects video and text features in the same embedding dimension. '
            f'Got video={video_features.shape[1]} text={text_features.shape[1]}.'
        )

    num_items = video_features.shape[0]
    train_idx, val_idx, test_idx = split_indices(num_items, args.val_ratio, args.test_ratio, seed)
    print('Split sizes train/val/test:', len(train_idx), len(val_idx), len(test_idx))

    train_dataset = torch.utils.data.TensorDataset(video_features[train_idx], text_features[train_idx])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = Proj_video(
        embedding_dim=video_features.shape[1],
        proj_dim=text_features.shape[1],
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        use_quantum=args.use_quantum,
    ).to(device)
    model.apply(weights_init_normal)
    logit_scale = nn.Parameter(torch.ones([], device=device) * np.log(1 / 0.07))
    optimizer = torch.optim.AdamW(itertools.chain(model.parameters(), [logit_scale]), lr=args.lr, betas=(0.5, 0.999))
    criterion_cls = nn.CrossEntropyLoss()

    best_val_loss = np.inf
    best_epoch = 0
    best_val_scores = None
    checkpoint_path = os.path.join(model_dir, 'video_text_Proj_video_quantum.pth' if args.use_quantum else 'video_text_Proj_video_classical.pth')
    history = []

    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for video_batch, text_batch in train_loader:
            if video_batch.shape[0] < 2:
                continue
            video_batch = video_batch.to(device)
            text_batch = text_batch.to(device)
            labels = torch.arange(video_batch.shape[0], device=device)

            video_embed = model(video_batch)
            text_embed = text_batch
            video_embed = video_embed / video_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)
            text_embed = text_embed / text_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)

            logits_per_video = logit_scale.exp() * video_embed @ text_embed.t()
            logits_per_text = logits_per_video.t()
            loss_video = criterion_cls(logits_per_video, labels)
            loss_text = criterion_cls(logits_per_text, labels)
            loss_cos = (loss_video + loss_text) / 2
            structure_loss = compute_structure_loss(video_batch, text_batch)
            loss = loss_cos + structure_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_batches += 1

        model.eval()
        with torch.no_grad():
            if len(val_idx) >= 2:
                val_video = video_features[val_idx].to(device)
                val_text = text_features[val_idx].to(device)
            else:
                val_video = video_features[train_idx].to(device)
                val_text = text_features[train_idx].to(device)
            val_labels = torch.arange(val_video.shape[0], device=device)
            val_video_embed = model(val_video)
            val_text_embed = val_text
            val_video_embed = val_video_embed / val_video_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)
            val_text_embed = val_text_embed / val_text_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)
            val_logits = logit_scale.exp() * val_video_embed @ val_text_embed.t()
            val_loss = (criterion_cls(val_logits, val_labels) + criterion_cls(val_logits.t(), val_labels)) / 2

        if val_loss.item() <= best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            if len(val_idx) >= 1:
                best_val_scores = evaluate_video_text_both_directions(model, video_features, text_features, val_idx, device)
            torch.save({
                'model': model.state_dict(),
                'logit_scale': logit_scale.detach().cpu(),
                'args': vars(args),
            }, checkpoint_path)

        if len(val_idx) >= 1:
            scores_v2t, scores_t2v = evaluate_video_text_both_directions(model, video_features, text_features, val_idx, device)
        else:
            scores_v2t, scores_t2v = evaluate_video_text_both_directions(model, video_features, text_features, train_idx, device)

        row = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss / max(epoch_batches, 1),
            'val_loss': val_loss.item(),
            'val_video_to_text_top1': scores_v2t['top1'],
            'val_video_to_text_top3': scores_v2t['top3'],
            'val_video_to_text_top5': scores_v2t['top5'],
            'val_text_to_video_top1': scores_t2v['top1'],
            'val_text_to_video_top3': scores_t2v['top3'],
            'val_text_to_video_top5': scores_t2v['top5'],
            'use_quantum': args.use_quantum,
        }
        history.append(row)
        print(
            'Epoch %03d train_loss=%.4f val_loss=%.4f val_v2t_top1=%.4f val_t2v_top1=%.4f' %
            (row['epoch'], row['train_loss'], row['val_loss'], row['val_video_to_text_top1'], row['val_text_to_video_top1'])
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if 'logit_scale' in checkpoint:
        logit_scale.data = checkpoint['logit_scale'].to(device)
    eval_idx = torch.arange(num_items) if args.eval_on_all else test_idx
    test_scores_v2t, test_scores_t2v = evaluate_video_text_both_directions(model, video_features, text_features, eval_idx, device)

    summary_row = {
        'epoch': 'best_test',
        'train_loss': np.nan,
        'val_loss': best_val_loss,
        'val_video_to_text_top1': best_val_scores[0]['top1'] if best_val_scores else np.nan,
        'val_video_to_text_top3': best_val_scores[0]['top3'] if best_val_scores else np.nan,
        'val_video_to_text_top5': best_val_scores[0]['top5'] if best_val_scores else np.nan,
        'val_text_to_video_top1': best_val_scores[1]['top1'] if best_val_scores else np.nan,
        'val_text_to_video_top3': best_val_scores[1]['top3'] if best_val_scores else np.nan,
        'val_text_to_video_top5': best_val_scores[1]['top5'] if best_val_scores else np.nan,
        'test_video_to_text_top1': test_scores_v2t['top1'],
        'test_video_to_text_top3': test_scores_v2t['top3'],
        'test_video_to_text_top5': test_scores_v2t['top5'],
        'test_text_to_video_top1': test_scores_t2v['top1'],
        'test_text_to_video_top3': test_scores_t2v['top3'],
        'test_text_to_video_top5': test_scores_t2v['top5'],
        'eval_on_all': args.eval_on_all,
        'use_quantum': args.use_quantum,
    }
    history.append(summary_row)

    result_csv = os.path.join(result_dir, 'video_text_results.csv')
    pd.DataFrame(history).to_csv(result_csv, index=False)
    print('The best epoch is:', best_epoch)
    print(
        'Best checkpoint test Top1/Top3/Top5: '
        'v2t %.6f/%.6f/%.6f, t2v %.6f/%.6f/%.6f' %
        (
            test_scores_v2t['top1'], test_scores_v2t['top3'], test_scores_v2t['top5'],
            test_scores_t2v['top1'], test_scores_t2v['top3'], test_scores_t2v['top5'],
        )
    )
    print('Saved results to:', result_csv)
    print('Saved checkpoint to:', checkpoint_path)
    return test_scores_v2t['top1'], test_scores_t2v['top1'], best_val_loss


def main():
    args = parser.parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.result_path = os.path.abspath(args.result_path)
    args.model_path = os.path.abspath(args.model_path)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    if args.task == 'video_text':
        train_video_text(args)
        return

    num_sub = args.num_sub   
    
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
    for i in range(num_sub):

        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(os.path.join(args.result_path, 'result.csv'))


class Print_model_info():
    def __init__(self):
        model_idx = 'test0'
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Proj_eeg.load_state_dict(torch.load(model_path + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load(model_path + model_idx + 'Proj_img_cls.pth'), strict=False)

        print(self.Proj_img[2].weight.shape)


def print_mdl_info():
    Print_model_info()



if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    # print_mdl_info()

    print(time.asctime(time.localtime(time.time())))
