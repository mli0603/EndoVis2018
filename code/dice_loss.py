import torch
import torch.nn as nn
import torch.nn.functional as F

class DICELoss(nn.Module):
    #DICE Loss Function

    def __init__(self, num_classes
                 use_gpu=True):
        super(DICELoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes

    def forward(self, scores, target):
        """DICE Loss
        Args:
            scores (tensor):  Predicted scores for every class on the image,
                shape: [batch_size,num_classes,w,h]
            targets (tensor): Ground truth labels,
                shape: [batch_size,]
        """
        number_of_classes = scores.shape[1]
        target_one_hot = torch.zeros_like(scores)
        target_one_hot.scatter_(1, target.view(scores.shape[0],1,scores.shape[2],scores.shape[3]), 1)
        smooth = 0.001
        loss = 0
        for cl in range(number_of_classes):
            iflat = scores[:,cl,:,:].contiguous().view(-1)
            tflat = target_one_hot[:,cl,:,:].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        return loss