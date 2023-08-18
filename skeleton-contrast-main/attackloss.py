import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import random
# from dataloaders import *
import argparse
import os
import sys
sys.path.append('../')

def boneLengthLoss(adData, refData):
    parents = np.array([0, 0, 20, 3, 20,
                        4, 5, 6, 20, 8,
                        9, 10, 0, 12, 13,
                        14, 0, 16, 17, 18,
                        1, 7, 7, 11, 11])

    # convert the data into shape (batchid, frameNo, jointNo, jointCoordinates)
    jpositions = torch.reshape(adData, (adData.shape[0],2, 300, 25, 3))

    jboneVecs = jpositions - jpositions[:,:, :, parents, :] + 1e-8

    boneLengths = torch.sqrt(torch.sum(torch.square(jboneVecs), axis=-1))

    positions = torch.reshape(refData, (refData.shape[0], 2, 300, 25, 3))

    boneVecs = positions - positions[:, :, :, parents, :] + 1e-8

    refBoneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

    boneLengthsLoss = torch.mean(torch.sum(torch.sum(torch.sum(torch.square(boneLengths - refBoneLengths), axis=-1), axis=-1), axis=-1))
    return boneLengthsLoss

def pairLengthLoss(adData, refData):
    parents = np.array([10, 0, 1, 2, 3,
                        10, 5, 6, 7, 8,
                        10, 10, 11, 12, 13,
                        13, 15, 16, 17, 18,
                        13, 20, 21, 22, 23])

    # convert the data into shape (batchid, frameNo, jointNo, jointCoordinates)
    jpositions = torch.reshape(adData, (adData.shape[0], 2, 300, 25, 3))

    jboneVecs = jpositions - jpositions[:, :, :, parents, :] + 1e-8

    boneLengths = torch.sqrt(torch.sum(torch.square(jboneVecs), axis=-1))

    positions = torch.reshape(refData, (refData.shape[0], 2, 300, 25, 3))

    boneVecs = positions - positions[:, :, :, parents, :] + 1e-8

    refBoneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))


    pairLengthsLoss= torch.mean(
        torch.sum(torch.sum(torch.sum(torch.square(boneLengths - refBoneLengths), axis=-1), axis=-1), axis=-1))
    return pairLengthsLoss

def accLoss (adData, refData, jointWeights ):
    deltaT=1/30
    # adData.cuda()
    # refData.cuda()
    # print(adData.size())
    refAcc = (refData[:,  2:, :] - 2 * refData[:,  1:-1, :] + refData[:,  :-2, :]) /deltaT / deltaT
    # print(refAcc.size())
    adAcc = (adData[:,  2:, :] - 2 * adData[:,  1:-1, :] + adData[:, :-2, :]) / deltaT / deltaT

    if jointWeights == None:
        return torch.mean(torch.sum(torch.sum(torch.square(adAcc - refAcc), axis=-1), axis=-1), axis=-1)
    else:
        adAcc = torch.reshape(adAcc, (adAcc.shape[0],2, adAcc.shape[1], 25, -1))
        refAcc = torch.reshape(refAcc, (refAcc.shape[0],2,  refAcc.shape[1], 25, -1))
        Accloss=torch.mean( torch.sum(torch.sum(torch.sum(torch.sum(torch.square(adAcc - refAcc), axis=-1)*jointWeights, axis=-1), axis=-1), axis=-1),axis=-1)
        return Accloss

def perceptualLoss(refData, adData, args):

    # the joint weights are decided per joint, the spinal joints have higher weights.
    # 和MSEloss结果不同，MSEloss在特征维度上也做了平均，我们只在batch维度上做平均
    # jointWeights = torch.Tensor([[[0.02, 0.02, 0.02, 0.02, 0.02,
    #                               0.02, 0.02, 0.02, 0.02, 0.02,
    #                               0.04, 0.04, 0.04, 0.04, 0.04,
    #                               0.02, 0.02, 0.02, 0.02, 0.02,
    #                               0.02, 0.02, 0.02, 0.02, 0.02]]]).cuda()
    jointWeights = torch.Tensor([[[0.04, 0.04, 0.04, 0.04, 0.02,
                                  0.02, 0.02, 0.02, 0.02, 0.02,
                                  0.02, 0.04, 0.02, 0.02, 0.02,
                                  0.02, 0.02, 0.02, 0.02, 0.02,
                                  0.02, 0.02, 0.04, 0.02, 0.02]]]).cuda()

    elements = args.perpLoss.split('_')
    refData = torch.reshape(refData, (refData.shape[0], 300, 25, 3, 2))
    refData = refData.permute(0, 3, 1, 2, 4)
    adData = torch.reshape(adData, (adData.shape[0], 300, 25, 3, 2))
    adData = adData.permute(0, 3, 1, 2, 4)
    adData = torch.reshape(adData, (adData.shape[0], 300, 150))
    refData = torch.reshape(refData, (refData.shape[0], 300, 150))
    # if elements[0] == 'l2' or elements[0] == 'l2Clip':

        # diffmx = K.square(refData - adData),
    squaredLoss = torch.sum(torch.reshape(torch.square(refData - adData), (refData.shape[0], 2,  300, 25, -1)),
                        axis=-1)

    weightedSquaredLoss = squaredLoss * jointWeights

    squareCost = torch.sum(torch.sum(torch.sum(weightedSquaredLoss, axis=-1), axis=-1),axis=-1)
    oloss = torch.mean(squareCost, axis=-1)

    if len(elements) == 1:
        return oloss

    elif elements[1] == 'bone':
        boneLengthsLoss = boneLengthLoss(refData, adData)
        pairloss = pairLengthLoss(refData, adData)

        return 0.5 * oloss + 0.5 * (0.5*boneLengthsLoss+0.5*pairloss)
        # return 0.7 * oloss + 0.3 * boneLengthsLoss
        # return 0.7 * oloss + 0.3 * boneLengthsLoss


    elif elements[1] == 'boneacc':
        boneLengthsLoss = boneLengthLoss(refData, adData)
        # pairloss = pairLengthLoss(refData, adData)
        jointAcc = accLoss(adData, refData, jointWeights)

        return 0.6 * oloss + 0.4 * (0.7*boneLengthsLoss+0.3*jointAcc)
        # return 0.7 * oloss + 0.3 * (0.9*pairloss+0.1*jointAcc)
    else:
        jointAcc = accLoss(adData, refData, jointWeights)

        return 0.8*oloss + 0.2*jointAcc