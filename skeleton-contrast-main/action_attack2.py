#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# from kmeans_pytorch import kmeans
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# from model import generate_model
# from models.resnet import get_fine_tuning_parameters
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score

from moco.GRU import *
from moco.HCN import HCN
from moco.AGCN import Model as AGCN
from attackloss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change for action recogniton
from dataset import get_finetune_training_set, get_finetune_validation_set

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50, 70, ], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--pretrained', default='./checkpoints/checkpoint_0450.pth.tar ', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')

parser.add_argument('--protocol', default='cross_view', type=str,
                    help='traiining protocol of ntu')

parser.add_argument('--finetune-skeleton-representation', default='graph-based', type=str,
                    help='which skeleton-representation to use for downstream training')
parser.add_argument('--pretrain-skeleton-representation', default='seq-based_and_graph-based', type=str,
                    help='which skeleton-representation where used for  pre-training')
parser.add_argument('--knn-neighbours', default=1, type=int,
                    help='number of neighbours used for KNN.')
parser.add_argument("-pl", "--perpLoss", required=False, help="to specify the perceptual loss",
                    default='l2_boneacc')
best_acc1 = 0
args = parser.parse_args()

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('PC weight initial finished!')


def load_moco_encoder_q(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def load_moco_encoder_r(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_r up to before the embedding layer
            if k.startswith('module.encoder_r') and not k.startswith('module.encoder_r.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_r."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def load_pretrained(args, model):
    # intra-skeleton contrastive  pretrianing
    if args.pretrain_skeleton_representation == 'seq-based' or args.pretrain_skeleton_representation == 'image-based' or args.pretrain_skeleton_representation == 'graph-based':

        # load  only seq-based / graph-based / image-based  query encoder  of  the intra-skeleton  framework pretrained using corresponding representation
        load_moco_encoder_q(model, args.pretrained)

    # inter-skeleton contrastive  pretrianing
    else:
        if args.finetune_skeleton_representation == 'seq-based' and (
                args.pretrain_skeleton_representation == 'seq-based_and_graph-based' or args.pretrain_skeleton_representation == 'seq-based_and_image-based'):
            # load  only seq-based query encoder of the inter-skeleton framework  pretrained using seq-based_and_graph-based or 'seq-based_and_image-based' representations
            load_moco_encoder_q(model, args.pretrained)

        elif args.finetune_skeleton_representation == 'graph-based' and args.pretrain_skeleton_representation == 'seq-based_and_graph-based':
            # load  only graph-based query encoder of the inter-skeleton framework pretrained using seq-based_and_graph-based representations
            load_moco_encoder_r(model, args.pretrained)

        elif args.finetune_skeleton_representation == 'graph-based' and args.pretrain_skeleton_representation == 'graph-based_and_image-based':
            # load  only graph-based query encoder of the inter-skeleton framework pretrained using graph-based_and_image-based representations
            load_moco_encoder_q(model, args.pretrained)

        elif args.finetune_skeleton_representation == 'image-based' and args.pretrain_skeleton_representation == 'seq-based_and_image-based':
            # load  only image-based query encoder of the inter-skeleton framework pretrained using seq-based_and_image-based representations
            load_moco_encoder_r(model, args.pretrained)

        elif args.finetune_skeleton_representation == 'image-based' and args.pretrain_skeleton_representation == 'graph-based_and_image-based':
            # load  only image-based query encoder of the inter-skeleton framework pretrained using graph-based_and_image-based representations
            load_moco_encoder_r(model, args.pretrained)


def knntraining(data_train, label_train, nn=9):
    label_train = np.asarray(label_train)
    print("Number of KNN Neighbours = ", nn)
    print("training feature and labels", data_train.shape, len(label_train))

    Xtr_Norm = preprocessing.normalize(data_train)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    return knn
def kmeanstraining(data_train, labels,  nc=60):
    print("Number of classes = ", nc)
    print("training feature and labels", data_train.shape)
    # scaler = preprocessing.MinMaxScaler()

    Xtr_Norm = preprocessing.normalize(data_train)
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(Xtr_Norm)

    return kmeans



def traindata_extract_hidden(model, data_train):
    for ith, (ith_data, label) in enumerate(data_train):
        # ith_data1 = torch.reshape(ith_data,(ith_data.shape[0], ith_data.shape[2], -1))
        input_tensor = ith_data.to(device)

        en_hi = model(input_tensor, knn_eval=True)
        en_hi = en_hi.squeeze()
        # print("encoder size",en_hi.size())

        if ith == 0:
            label_train = label
            hidden_array_train = en_hi[:, :].detach().cpu().numpy()

        else:
            label_train = np.hstack((label_train, label))
            hidden_array_train = np.vstack((hidden_array_train, en_hi[:, :].detach().cpu().numpy()))




        # #
        # if ith > 2000: # for debug
        #     break

    return hidden_array_train,  label_train

def attackbyknn(knn, model, data_eval):
    model.eval()
    with torch.no_grad():
        for ith, (ith_data, label) in enumerate(data_eval):
            input_tensor = ith_data.to(device)
            # scaler = preprocessing.MinMaxScaler()


            en_hi = model(input_tensor, knn_eval=True)
            en_hi = en_hi.squeeze()
            features = en_hi.detach().cpu().numpy()
            Xte_Norm = preprocessing.normalize(features)
            pred = knn.predict(Xte_Norm)



            if ith == 0:
                hidden_array_eval = en_hi[:, :].detach().cpu().numpy()
                label_eval = label
            else:
                label_eval = np.hstack((label_eval, label))
                hidden_array_eval = np.vstack((hidden_array_eval, en_hi[:, :].detach().cpu().numpy()))

    return  hidden_array_eval,  label_eval


def attackbykmeans( model, data_eval,hi_train):
    for ith, (ith_data, label) in enumerate(data_eval):
        hardsamples = findhardsamples(hi_train, model, ith_data, label)
        # 参数设置
        # torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = False

        prednew1 = []
        input_tensor = ith_data.to(device)
        input = input_tensor.clone()
        momentum = torch.zeros_like(input).detach().cuda()
        c = torch.zeros((8, 3, 1, 25, 2)).cuda()
        ep = 400
        # hi_train = hi_train[1200:1600]

        # 找中心
        # with torch.no_grad():
        #     model.eval()
        #     ep = 500
        #     en_hi = model(input_tensor, knn_eval=True)
        #     en_hi = en_hi.squeeze()
        # features = en_hi.detach().cpu().numpy()
        # 找第二近的点
        # precen2 = features.copy()
        # 按排序其他所有samples
        # for i in range(len(label)):
        #     # hi_train[0]=precen2[i]
        #     # x = hi_train.copy()
        #     precen2[[0,i]]= precen2[[i,0]]
        #     x = precen2.copy()
        #     precen2 = features.copy()
        #     prednew1.append(x)
        # pred1 = np.array(prednew1)

        # for i in range(len(label)):
        #     precen1[i,:] = cluscen[pred1[i],:]
        input.requires_grad = True
        # precen1 = torch.tensor(pred1).cuda()
        eps = 0.01
        # 加入初始扰动
        # input= input + input.new(input.size()).uniform_(-0.003, 0.003)
        for i in range(ep):
            input_old = input.clone()
            model.eval()
            en_hiatt = model(input, knn_eval=True)
            en_hiatt = en_hiatt.squeeze()
            # en_hiatt = torch.nn.functional.normalize(en_hiatt, p=2, dim=1)
            en_hiatt = en_hiatt.unsqueeze(1).repeat(1,51,1)
            cos = nn.CosineSimilarity(dim=-1, eps=1e-1)
            # simloss = cos(en_hiatt, precen1)
            simloss = cos(en_hiatt, hardsamples)
            # closs = nn.MSELoss(reduction='none')(en_hiatt, precen).sum(dim = -1)
            # closs = 1*torch.sum(simloss)
            # ploss = perceptualLoss(input_tensor, input, args)
            label = torch.tensor([0] * len(label)).long().cuda()
            loss = -1*nn.CrossEntropyLoss()(simloss, label)
            # loss = closs
            input.grad = None
            input.retain_grad()
            loss.backward(retain_graph=True)
            cgs = input.grad
            cgs = cgs / torch.mean(torch.abs(cgs), dim=(1, 2, 3, 4), keepdim=True)
            change = cgs[:, :, 1:, :, :] - cgs[:, :, :-1, :, :]
            changes = torch.cat((c, change), dim=2)
            cgs = cgs + momentum * 1 - 0.1 * changes
            momentum = cgs
            cgs = cgs.sign()
            input = input - 1. / 10000 * cgs
            # loss.backward()
            # input = torch.reshape(input, (input.shape[0], input.shape[1], 75, 2))
            # input_tensor = torch.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], 75, 2))
            for k in range(len(label)):
                double = torch.nonzero(input_tensor[k, :, :, :, 1])
                if double.size() == torch.Size([0, 3]):
                    input[k, :, :, :, 1] = input_tensor[k, :, :,:,  1]
            # for k in range(len(label)):
            #     double = torch.nonzero(input_tensor[k, :, :,  1])
            #     if double.size() == torch.Size([0, 2]):
            #         input[k, :, :,  1] = input_tensor[k, :, :, 1]
            # input = torch.reshape(input, (input.shape[0], input.shape[1], 150))
            # input_tensor = torch.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], 150))
            if i % 25 == 0 or i == 0 or i == ep - 1:
                print(loss,  i)
                np.savez_compressed('./Samples/NTU%dsamples.npz' % (ith),
                                    clips=ith_data.cpu().detach().numpy(), oriClips=input.cpu().detach().numpy(),
                                    labels=label.cpu().detach().numpy())
            # Lp norm:
            input = torch.where(input > input_tensor + eps, input_tensor + eps, input)
            input = torch.where(input < input_tensor - eps, input_tensor - eps, input)

            # L2 norm
            # updateall = input - input_tensor
            # updates = input - input_old
            # for ci in range(len(label)):
            #     updateNorm = torch.sqrt(torch.sum(torch.sum(torch.square(updateall[ci]))))
            #     if  updateNorm > 1.5:
            #         updates[ci] = 0
            # input = input_old+updates
            # a=1
            # for ci in range(len(label)):
            #     updateNorm = torch.sqrt(torch.sum(torch.sum(torch.square(updates[ci]))))
            #     if  updateNorm > 1.5:
            #         updates[ci] = updates[ci]/2
            # input = input_tensor+updates


        # with torch.no_grad():
        #     en_hiatt1 = model(input, knn_eval=True)
        #     en_hiatt1 = en_hiatt1.squeeze()
        #     features = en_hiatt1.detach().cpu().numpy()
        #     Xta_Norm = preprocessing.normalize(features)
        #     preda = kmeans.predict(Xta_Norm)
            # updates = input - input_old
            # sucIndices = []
            # for i in range(len(label)):
            #     if preda[i] != pred[i]:
            #         sucIndices.append(i)
            # updates[sucIndices] = torch.zeros((300, 150)).cuda()
            # input = input_old+updates



        # model.eval()








    return  input

class MyAutoDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # self.xy = zip(self.data, self.label)

    def __getitem__(self, index):
        sequence = self.data[index, :]
        label = self.label[index]
        # Transform it to Tensor
        # x = torchvision.transforms.functional.to_tensor(sequence)
        # x = torch.tensor(sequence, dtype=torch.float)
        # y = torch.tensor([self.label[index]], dtype=torch.int)

        return sequence, label

    def __len__(self):
        return len(self.label)


def train_autoencoder(hidden_train, hidden_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches):
    batch_size = 64
    # auto = autoencoder(hidden_train.shape[1], middle_size).to(device)
    auto = autoencoder(hidden_train.shape[1], middle_size).cuda()
    auto_optimizer = optim.Adam(auto.parameters(), lr=0.001)
    auto_scheduler = optim.lr_scheduler.LambdaLR(auto_optimizer, lr_lambda=lambda1)
    criterion_auto = nn.MSELoss()

    autodataset = MyAutoDataset(hidden_train, label_train)
    trainloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

    autodataset = MyAutoDataset(hidden_eval, label_eval)
    evalloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoches):
        for (data, label) in trainloader:
            # img, _ = data
            # img = img.view(img.size(0), -1)
            # img = Variable(img).cuda()
            # data = torch.tensor(data.clone().detach(), dtype=torch.float).to(device)
            # ===================forward=====================
            data = data.cuda()
            output, _ = auto(data)
            loss = criterion(output, data)
            # ===================backward====================
            auto_optimizer.zero_grad()
            loss.backward()
            auto_optimizer.step()
            auto_scheduler.step()
        # ===================log========================
        for (data, label) in evalloader:
            data = data.cuda()
            # ===================forward=====================
            output, _ = auto(data)
            loss_eval = criterion(output, data)
        # if epoch % 200 == 0:
        #   print('epoch [{}/{}], train loss:{:.4f} eval loass:{:.4f}'
        #         .format(epoch + 1, num_epoches, loss.item(), loss_eval.item()))

    ## extract hidden train
    count = 0
    for (data, label) in trainloader:
        data = data.cuda()
        _, encoder_output = auto(data)

        if count == 0:
            np_out_train = encoder_output.detach().cpu().numpy()
            label_train = label
        else:
            label_train = np.hstack((label_train, label))
            np_out_train = np.vstack((np_out_train, encoder_output.detach().cpu().numpy()))
        count += 1

    ## extract hidden eval
    count = 0
    for (data, label) in evalloader:
        data = data.cuda()
        _, encoder_output = auto(data)

        if count == 0:
            np_out_eval = encoder_output.detach().cpu().numpy()
            label_eval = label

        else:
            label_eval = np.hstack((label_eval, label))
            np_out_eval = np.vstack((np_out_eval, encoder_output.detach().cpu().numpy()))
        count += 1

    return np_out_train, np_out_eval, label_train, label_eval


class autoencoder(nn.Module):
    def __init__(self, input_size, middle_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, middle_size),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(middle_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_size),
        )

    def forward(self, x):
        middle_x = self.encoder(x)
        x = self.decoder(middle_x)
        return x, middle_x

def findhardsamples(hi_train, model, input_tensor,label):
    hi_train= torch.tensor(hi_train).cuda()
    # X = hi_train.shape[0]//30
    tar_mot = []
    input_tensor =  input_tensor.to(device)
    with torch.no_grad():
        model.eval()
        en_hi = model(input_tensor, knn_eval=True)
        en_hi = en_hi.squeeze()
    for i in  range(len(label)):
        # 剔除相同标签
        la = label[i]
        hard_samples = []
        a = en_hi[i].repeat(hi_train.shape[0],1)
        MSE=torch.sum(torch.square(a-hi_train),dim=-1)
        sorted, indices = torch.sort(MSE)
        wholeset=indices[0:0+500]
        for q in range(500):
            if (wholeset[q]-la)%60 != 0:
                hard_samples.append(wholeset[q])
        # hardsamples = torch.tensor(hard_samples)

        hardsamples = hi_train[hard_samples[0:50]]
        tar_mot.append(hardsamples)
    tar_mot = torch.reshape(torch.cat(tar_mot, dim=0),(8,-1,256))
    hardsample = torch.cat((en_hi.unsqueeze(1), tar_mot), dim=1)



    return  hardsample

def clustering_knn_acc(model, train_loader, eval_loader, criterion, num_epoches=400, middle_size=125, knn_neighbours=5):
    model.eval()
    # find hard samples
    hi_train,  label_train = traindata_extract_hidden(model, eval_loader)
    # knnmodel = knntraining(hi_train, label_train, nn=knn_neighbours)
    # kmeansmodel = kmeanstraining(hi_train, label_train, nc=60)
    # result = attackbykmeans(kmeansmodel, model, eval_loader)
    result = attackbykmeans( model, eval_loader,hi_train)
    # result = attackbyknn(knnmodel, model, eval_loader)
    # print(hi_train.shape)

    # lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)
    # np_out_train, np_out_eval, au_l_train, au_l_eval = train_autoencoder(hi_train, hi_eval, label_train,
    #                                                                      label_eval, middle_size, criterion, lambda1,
    #                                                                      num_epoches)

    # print(hi_train.shape)

    # knn_acc_au = knn(np_out_train, np_out_eval, au_l_train, au_l_eval, nn=knn_neighbours)
    return result
    # return knn_acc_1, knn_acc_au


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # training dataset
    from options import options_retrieval as options
    if args.finetune_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.finetune_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.finetune_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.finetune_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()

    opts.train_feeder_args['input_representation'] = args.finetune_skeleton_representation
    opts.test_feeder_args['input_representation'] = args.finetune_skeleton_representation

    # create model
    if args.finetune_skeleton_representation == 'seq-based':
        # Gru model
        model = BIGRU(**opts.bi_gru_model_args)
        print(model)
        print("options", opts.bi_gru_model_args, opts.train_feeder_args, opts.test_feeder_args)
        if not args.pretrained:
            weights_init_gru(model)

    elif args.finetune_skeleton_representation == 'graph-based':
        model = AGCN(**opts.agcn_model_args)
        print(model)
        print("options", opts.agcn_model_args, opts.train_feeder_args, opts.test_feeder_args)

    elif args.finetune_skeleton_representation == 'image-based':
        model = HCN(**opts.hcn_model_args)
        print(model)
        print("options", opts.bi_gru_model_args, opts.train_feeder_args, opts.test_feeder_args)

    if args.pretrained:
        # freeze all layers
        for name, param in model.named_parameters():
            param.requires_grad = False

    # load from pre-trained  model
    load_pretrained(args, model)

    if args.gpu is not None:
        model = model.cuda()
        # torch.cuda.empty_cache()
        # model = nn.DataParallel(model, device_ids=None)

    cudnn.benchmark = True

    ## Data loading code

    train_dataset = get_finetune_training_set(opts)
    val_dataset = get_finetune_validation_set(opts)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    auto_criterion = nn.MSELoss()
    # Extract frozen features of  the  pre-trained query encoder
    # train and evaluate a KNN  classifier on extracted features
    acc1 = clustering_knn_acc(model, train_loader, val_loader, criterion=auto_criterion,
                                      knn_neighbours=args.knn_neighbours)

    print(" Knn Without AE= ", acc1, " Knn With AE=", acc_au)


if __name__ == '__main__':
    main()
