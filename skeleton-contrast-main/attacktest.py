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
from sklearn import preprocessing
from sklearn.cluster import KMeans
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# from model import generate_model
# from models.resnet import get_fine_tuning_parameters
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from moco.GRU import *
from moco.HCN import HCN
from moco.AGCN import Model as AGCN

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
parser.add_argument('-b', '--batch-size', default=16, type=int,
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
parser.add_argument('--resume', default='./checkpoints/seq-based_checkpoint.pth.tar ', type=str, metavar='PATH',
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

parser.add_argument('--finetune-skeleton-representation', default='seq-based', type=str,
                    help='which skeleton-representation to use for downstream training')
parser.add_argument('--pretrain-skeleton-representation', default='seq-based', type=str,
                    help='which skeleton-representation where used for  pre-training')

best_acc1 = 0


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

        # fine tune only seq-based / graph-based / image-based  query encoder  of  the intra-skeleton  framework pretrained using corresponding representation
        load_moco_encoder_q(model, args.pretrained)
        finetune_encoder_q = True
        finetune_encoder_r = False
        return finetune_encoder_q, finetune_encoder_r

    # inter-skeleton contrastive  pretrianing
    else:
        if args.finetune_skeleton_representation == 'seq-based' and (
                args.pretrain_skeleton_representation == 'seq-based_and_graph-based' or args.pretrain_skeleton_representation == 'seq-based_and_image-based'):
            # fine tune only seq-based query encoder of the inter-skeleton framework  pretrained using seq-based_and_graph-based or 'seq-based_and_image-based' representations
            load_moco_encoder_q(model, args.pretrained)
            finetune_encoder_q = True
            finetune_encoder_r = False

        elif args.finetune_skeleton_representation == 'graph-based' and args.pretrain_skeleton_representation == 'seq-based_and_graph-based':
            # fine tune only graph-based query encoder of the inter-skeleton framework pretrained using seq-based_and_graph-based representations
            load_moco_encoder_r(model, args.pretrained)
            finetune_encoder_q = False
            finetune_encoder_r = True

        elif args.finetune_skeleton_representation == 'graph-based' and args.pretrain_skeleton_representation == 'graph-based_and_image-based':
            # fine tune only graph-based query encoder of the inter-skeleton framework pretrained using graph-based_and_image-based representations
            load_moco_encoder_q(model, args.pretrained)
            finetune_encoder_q = True
            finetune_encoder_r = False

        elif args.finetune_skeleton_representation == 'image-based' and args.pretrain_skeleton_representation == 'seq-based_and_image-based':
            # fine tune only image-based query encoder of the inter-skeleton framework pretrained using seq-based_and_image-based representations
            load_moco_encoder_r(model, args.pretrained)
            finetune_encoder_q = False
            finetune_encoder_r = True

        elif args.finetune_skeleton_representation == 'image-based' and args.pretrain_skeleton_representation == 'graph-based_and_image-based':
            # fine tune only image-based query encoder of the inter-skeleton framework pretrained using graph-based_and_image-based representations
            load_moco_encoder_r(model, args.pretrained)
            finetune_encoder_q = False
            finetune_encoder_r = True

        return finetune_encoder_q, finetune_encoder_r


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

    # create model

    # training dataset
    from options import options_classification as options
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

    if args.finetune_skeleton_representation == 'seq-based':
        # Gru model
        model = BIGRU(**opts.bi_gru_model_args)
        print(model)
        print("options", opts.bi_gru_model_args, opts.train_feeder_args, opts.test_feeder_args)
        # if not args.pretrained:
        #     weights_init_gru(model)

    elif args.finetune_skeleton_representation == 'graph-based':
        model = AGCN(**opts.agcn_model_args)
        print(model)
        print("options", opts.agcn_model_args, opts.train_feeder_args, opts.test_feeder_args)

    elif args.finetune_skeleton_representation == 'image-based':
        model = HCN(**opts.hcn_model_args)
        print(model)
        print("options", opts.bi_gru_model_args, opts.train_feeder_args, opts.test_feeder_args)


    # load from pre-trained  model

    # finetune_encoder_q, finetune_encoder_r = load_pretrained(args, model)

    if args.gpu is not None:
        model = model.cuda()


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    ## Data loading code

    val_dataset = get_finetune_validation_set(opts)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    for epoch in range(args.start_epoch, args.epochs):


        # adjust_learning_rate(optimizer, epoch, args)
        #
        # # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch, args)

        attack(val_loader, model, args)

        # sanity check
        # if epoch == args.start_epoch:
        #     if finetune_encoder_q:
        #         sanity_check_encoder_q(model.state_dict(), args.pretrained)
        #     elif finetune_encoder_r:
        #         sanity_check_encoder_r(model.state_dict(), args.pretrained)
    print("Final  best accuracy", best_acc1)




def attack(val_loader, model, args):


    # switch to evaluate mode
    torch.backends.cudnn.enabled = False
    model.eval()

    # with torch.no_grad():
    # end = time.time()
    for i, (input_tensor, target) in enumerate(val_loader):
        if args.gpu is not None:
            input_tensor = input_tensor.cuda()
            input = input_tensor.clone()
            target = target.cuda()
            # target = target+1
        input.requires_grad = True
        ep = 400
        eps= 0.01
        model.eval()
        output1 = model(input_tensor)
        attacklabels1 = torch.argmax(output1, axis=1)
        print(attacklabels1)
        # compute output
        for p in range(ep):
            model.eval()
            output = model(input)
            attacklabels = torch.argmax(output, axis=1)
            classLoss = torch.nn.CrossEntropyLoss()(output, attacklabels)
            loss = -1*classLoss
            input.grad = None
            input.retain_grad()
            loss.backward(retain_graph=True)
            cgs = input.grad
            dd = torch.nonzero(cgs)
            cgs = cgs.sign()
            input = input - 1. / 10000 * cgs
            input = torch.reshape(input, (input.shape[0], input.shape[1], 75, 2))
            input_tensor = torch.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], 75, 2))
            for k in range(len(target)):
                double = torch.nonzero(input_tensor[k, :, :, 1])
                if double.size() == torch.Size([0, 2]):
                    input[k, :, :,  1] = input_tensor[k, :, :,  1]
            input = torch.reshape(input, (input.shape[0], input.shape[1], 150))
            input_tensor = torch.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], 150))
            if p % 25 == 0 or p == 0 or p == ep - 1:
                print(loss,  p)
                np.savez_compressed('./Samples/NTU%dfit.npz' % (i),
                                    clips=input_tensor.cpu().detach().numpy(), oriClips=input.cpu().detach().numpy(),
                                    labels=attacklabels.cpu().detach().numpy())
            # Lp norm:
            input = torch.where(input > input_tensor + eps, input_tensor + eps, input)
            input = torch.where(input < input_tensor - eps, input_tensor - eps, input)
            updates = input - input_tensor




    return input



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + 'model_best.pth.tar')


def sanity_check_encoder_q(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def sanity_check_encoder_r(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_r.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_r.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")












if __name__ == '__main__':
    main()
