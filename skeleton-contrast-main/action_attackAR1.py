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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from moco.GRU import *
from moco.HCN import HCN
from moco.AGCN import Model as AGCN
from attackloss import *
from feeder.augmentations import  *



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

# parser.add_argument('--pretrained', default='./checkpoints/interskeleton_seq_based_graph_based/checkpoint_0450.pth.tar', type=str,
#                     help='path to moco pretrained checkpoint')
parser.add_argument('--pretrained', default='./checkpoints/checkpoint_test_0450.pth.tar', type=str,
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
            # if k.startswith('module.encoder_r') and not k.startswith('module.encoder_r.fc'):
            #     # remove prefix
            #     state_dict[k[len("module.encoder_r."):]] = state_dict[k]
            # # delete renamed or unused k
            # del state_dict[k]
            if k.startswith('encoder_r') and not k.startswith('encoder_r.fc'):
                # remove prefix
                state_dict[k[len("encoder_r."):]] = state_dict[k]
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
def kmeanstraining(data_train, labels,  nc=150):
    print("Number of classes = ", nc)
    print("training feature and labels", data_train.shape)
    # scaler = preprocessing.MinMaxScaler()

    Xtr_Norm = preprocessing.normalize(data_train)
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(Xtr_Norm)

    return kmeans



def traindata_extract_hidden(model, data_train):
    for ith, (ith_data, label, number_of_frame) in enumerate(data_train):
        # ith_data1 = torch.reshape(ith_data,(ith_data.shape[0], ith_data.shape[2], -1))
        input_tensor = ith_data.to(device)

        en_hi = model(input_tensor, knn_eval=True)
        en_hi = en_hi.squeeze()
        # print("encoder size",en_hi.size())

        if ith == 0:
            label_train = label
            hidden_array_train = en_hi[:, :].detach().cpu().numpy()

        else:
            label_traiqn = np.hstack((label_train, label))
            hidden_array_train = np.vstack((hidden_array_train, en_hi[:, :].detach().cpu().numpy()))
        # # # # #
        # if ith > 20: # for debug
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
def aug(input):
    input = input.cpu().detach().numpy()
    a = torch.rand(1)
    for r in range(8):
        if a<0.5:
            input[r] = pose_augmentation(input[r])
        elif a<1:
            input[r] = joint_courruption(input[r])
        else:
            input[r] = input[r]
    input = torch.from_numpy(input).cuda()
    return input
def  attackbykmeans( kmeans, model, data_eval):
    for ith, (ith_data, label, number_of_frame) in enumerate(data_eval):
        # if ith > 300:
        #     break
        # if ith < 499:
        #     continue
        # 参数设置
        # torch.backends.cudnn.enabled = False
        model.eval()
        prednew1 = []
        prednew2 = []
        input_tensor = ith_data.to(device)
        input = input_tensor.clone()
        momentum = torch.zeros_like(input).detach().cuda()

        # 找中心
        ep = 400
        en_hi = model(input_tensor, knn_eval=True)
        en_hi = en_hi.squeeze()
        features = en_hi.detach().cpu().numpy()
        Xte_Norm = preprocessing.normalize(features)
        # pred = kmeans.predict(Xte_Norm)
        cluscen = kmeans.cluster_centers_
        # 找第二近的点
        precen1 = np.zeros((8,110,256))
        precen2 = Xte_Norm.copy()

        # 按排序其他所有samples
        for i in range(len(label)):
            # hi_train[0]=precen2[i]
            # x = hi_train.copy()
            precen2[[0, i]] = precen2[[i, 0]]
            x = precen2.copy()
            precen2 = Xte_Norm.copy()
            prednew2.append(x)
        pred2 = np.array(prednew2)
        # 按排序找到所有中心
        for i in range(len(label)):
            a = np.expand_dims(precen2[i, :], 0).repeat(120,axis=0)
            idx = (np.sqrt(np.sum(np.square(a - cluscen), axis=-1)))

            max_index1 = np.argsort(idx)
            max_index1 = max_index1[10:]
            prednew1.append(max_index1)
        pred1 = np.array(prednew1)

        for i in range(len(label)):
            precen1[i, :] = cluscen[pred1[i], :]

        precen1 = torch.tensor(precen1)
        precen2 = torch.tensor(pred2)
        precencat = torch.cat((precen2, precen1), dim=1).cuda()
        eps = 0.01
        c = torch.zeros((8, 3, 1, 25, 2)).cuda()
        path = '../../AR1model/NTU{}CV1val.npz'.format(ith)
        data = np.load(path)
        Ar = data['AR']
        # Ar2 = data['AR2']
        # vel, vel1 = autore(input, number_of_frame, label)
        vel = torch.from_numpy(Ar).cuda()
        # vel1 = torch.from_numpy(Ar1).cuda()
        input.requires_grad = True
        for i in range(ep):
            input_old = input.clone()
            # input_ts=aug(input)
            # input_ts.requires_grad = True
            model.eval()
            # en_hiatt = model(input, knn_eval=True)
            en_hiatt = model(input, knn_eval=True)
            # precencat = findnewcentre(en_hiatt, kmeans, precen2)
            en_hiatt = en_hiatt.squeeze()
            en_hiatt = torch.nn.functional.normalize(en_hiatt, p=2, dim=1)
            en_hiatt = en_hiatt.unsqueeze(1).repeat(1,118,1)
            cos = nn.CosineSimilarity(dim=-1, eps=1e-1)
            simloss = cos(en_hiatt, precencat)

            # loss1 =  (F.log_softmax(simloss, dim=1) * pos_mask).sum(1) / pos_mask.sum(1)
            # loss = loss1.mean()
            # s = simloss.cpu().detach().numpy()
            # closs = nn.MSELoss(reduction='none')(en_hiatt, precen).sum(dim = -1)
            # closs = 1*torch.sum(simloss)
            # ploss = perceptualLoss(input_tensor, input_ts, args)
            labels = torch.tensor([0] * len(label)).long().cuda()
            loss = -1*nn.CrossEntropyLoss()(simloss, labels)
            # loss=loss1
            # loss = closs
            input.grad = None
            input.retain_grad()
            loss.backward(retain_graph=True)
            cgs = input.grad
            # input_ts.grad = None
            # input_ts.retain_grad()
            # loss.backward(retain_graph=True)
            # cgs = input_ts.grad
            cgs = cgs / torch.mean(torch.abs(cgs), dim=(1,2,3,4), keepdim=True)
            # buffer = cgs
            ## 计算平均速度
            # change = cgs[:, :, 1:,:, :] - cgs[:, :, :-1, :, :]
            # acc=  cgs[:, :, 2:,:, :] - 2*cgs[:, :, 1:-1, :, :]+cgs[:, :, :-2, :, :]
            # accs = torch.cat((c,c, acc),dim=2)
            change = cgs[:, :, 1:, :, :]
            # change = cgs[:, :, :-1, :, :]
            changes = torch.cat((change,c),dim=2)
            # change1 = cgs[:, :, 2:, :, :]
            # change = cgs[:, :, :-1, :, :]
            # changes1 = torch.cat((change1,c,c),dim=2)
            # vel = velcal(input, vel).cuda()
            # cgs = cgs + momentum*1+0.3*changes
            cgs = cgs + momentum * 1 + vel * changes
            # cgs = cgs + momentum * 1 + vel1 * changes + vel * changes1
            # cgs = cgs  + vel1 * changes + vel * changes1
            # cgs = cgs + momentum * 1
            # momentum = cgs


            cgs = cgs.sign()
            input = input - 1./10000 * cgs

            for k in range(len(label)):
                double = torch.nonzero(input_tensor[k, :, :, :, 1])
                if double.size() == torch.Size([0, 3]):
                    input[k, :, :, :, 1] = input_tensor[k, :, :,:,  1]
            # input = torch.reshape(input, (input.shape[0], input.shape[1], 150))
            # input_tensor = torch.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], 150))
            input = input.to(torch.float32)
            input = torch.where(input > input_tensor + eps, input_tensor + eps, input)
            input = torch.where(input < input_tensor - eps, input_tensor - eps, input)
            if i % 25 == 0 or i == 0 or i == ep - 1:
                print(loss,  i)
            # if i<=20:
            #     np.savez_compressed('./Samples/NTU%dAR2gcnidml21%d.npz' % (ith,i),
            #                         clips=ith_data.cpu().detach().numpy(), oriClips=input.cpu().detach().numpy(),
            #                         labels=label.cpu().detach().numpy())
            # if i >20:
            #     break
            if i == ep-1:
                np.savez_compressed('./SamplestestS1I-FGSM/NTU%dgcntests1i1.npz' % (ith),
                                    clips=ith_data.cpu().detach().numpy(), oriClips=input.cpu().detach().numpy(),
                                    labels=label.cpu().detach().numpy())
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








    return  input



def clustering_knn_acc(model, train_loader, eval_loader, criterion, num_epoches=400, middle_size=125, knn_neighbours=5):
    model.eval()
    # hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)
    # hi_train,  label_train = traindata_extract_hidden(model, train_loader)
    hi_eval, label_eval = traindata_extract_hidden(model, eval_loader)
    # hi_total= np.concatenate((hi_train, hi_eval),axis=0)
    # knnmodel = knntraining(hi_train, label_train, nn=knn_neighbours)
    kmeansmodel = kmeanstraining(hi_eval, label_eval, nc=120)
    # result = attackbykmeans(kmeansmodel, model, eval_loader)
    result = attackbykmeans(kmeansmodel, model, eval_loader)

    return result



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



if __name__ == '__main__':
    main()
