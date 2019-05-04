import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def DICE(model, test_dataloader, num_classes, use_gpu):
  #A false positive is a result that indicates a given condition exists, when it does not
  #A false negative is a test result that indicates that a condition does not hold, while in fact it does
    TP = np.zeros(num_classes).astype(np.int)
    FP = np.zeros(num_classes).astype(np.int)
    FN = np.zeros(num_classes).astype(np.int)
    dice = np.zeros(num_classes)
    model.eval()
    for data in test_dataloader:
        img, target = data
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        out = model(img)
        predict = torch.argmax(out, dim = 1)
        for cl in range(num_classes):
            gt_mask = (target == cl)
            pred_mask = (predict == cl)
            TP[cl] += (predict[gt_mask] == cl).sum()
            FP[cl] += (target[pred_mask] != cl).sum()
            FN[cl] += (predict[gt_mask] != cl).sum()
            print("class:",cl,":",TP[cl], FP[cl], FN[cl])
        for cl in range(num_classes):
            if(2*TP[cl] + FP[cl] + FN[cl] == 0):
              dice[cl] = -1
            else:
              dice[cl] = 2*TP[cl]/(2*TP[cl] + FP[cl] + FN[cl])
    m_dice = dice[dice>=0].mean()
    return m_dice

def IoU(model, test_dataloader, num_classes, use_gpu):
    TP = np.zeros(num_classes).astype(np.int)
    FP = np.zeros(num_classes).astype(np.int)
    FN = np.zeros(num_classes).astype(np.int)
    iou = np.zeros(num_classes)
    model.eval()
    for data in test_dataloader:
        img, target = data
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        out = model(img)
        predict = torch.argmax(out, dim = 1)
        for cl in range(num_classes):
            gt_mask = (target == cl)
            pred_mask = (predict == cl)
            TP[cl] += (predict[gt_mask] == cl).sum()
            FP[cl] += (target[pred_mask] != cl).sum()
            FN[cl] += (predict[gt_mask] != cl).sum()
            print("class:",cl,":",TP[cl], FP[cl], FN[cl])
        for cl in range(num_classes):
            if(TP[cl] + FP[cl] + FN[cl] == 0):
              iou[cl] = -1
            else:
              iou[cl] = TP[cl]/(TP[cl] + FP[cl] + FN[cl])
    m_iou = iou[iou>=0].mean()
    return m_iou