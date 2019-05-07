import torch
import torch.nn as nn
import torch.nn.functional as F

class DICELoss(nn.Module):
    #DICE Loss Function

    def __init__(self, weights):
        #weights(tensor): weights for every class when calculating dice loss
        super(DICELoss, self).__init__()
        self.weights = weights

    def forward(self, scores, target):
        """DICE Loss
        Args:
            scores (tensor):  Predicted scores for every class on the image,
                shape: [batch_size,num_classes,w,h]
            targets (tensor): Ground truth labels,
                shape: [batch_size,]
        """
        scores = F.softmax(scores, dim=1)
        number_of_classes = scores.shape[1]
        target_one_hot = torch.zeros_like(scores)
        target_one_hot.scatter_(1, target.view(scores.shape[0],1,scores.shape[2],scores.shape[3]), 1)
        smooth = 1e-7
        loss = 0
        for cl in range(number_of_classes):
            iflat = scores[:,cl,:,:].contiguous().view(-1)
            tflat = target_one_hot[:,cl,:,:].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            loss += (1 - ((2. * intersection) / (iflat.sum() + tflat.sum() + smooth)))*self.weights[cl]
        return loss/self.weights.sum(), scores, target_one_hot
    
    
# define dice loss function
def diceloss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    
    return (1 - dice_loss), probas, true_1_hot    

def label_accuracy(probas, true_1_hot):
    """Computes the accuracy.
    Args:
        probas: a tensor of shape [B, C, H, W] of probabilities
        true_1_hot: a tensor of shape [B, C, H, W]. Corresponds to the true label
    Returns:
        tp: [C] true positive of c classes
        fp: [C] false positive
        fn: [C] false negative
    """
    num_class = probas.shape[1]
    num_batch = probas.shape[0]
    
    pred = torch.max(probas,dim=1)[1]
    pred_1_hot = torch.eye(num_class)[pred.squeeze(1)]
    pred_1_hot = pred_1_hot.permute(0, 3, 1, 2).float()
    
    # sum all except class axis
    tp = torch.mul(pred_1_hot, true_1_hot).sum(dim=3).sum(dim=2).sum(dim=0)
    fp = pred_1_hot.sum(dim=3).sum(dim=2).sum(dim=0) - tp
    fn = true_1_hot.sum(dim=3).sum(dim=2).sum(dim=0) - tp
    
    return tp, fp, fn

if __name__ == "__main__":
    x = torch.rand([1,3,2,2])
    gt = torch.tensor([[[1,2],[2,0]]])
    weights = torch.tensor([1.0,1,1])
    DICE = DICELoss(weights)
    print(diceloss(x, gt)[0])#, diceloss(x, gt)[1], diceloss(x, gt)[2])
    print(DICE(x, gt)[0])#, DICE(x, gt)[1], DICE(x, gt)[2])
# # test functions
# x = torch.tensor([[[0.1,0.2],[0.3,0.4]],[[0.2,0.3],[0.3,0.4]],[[0.3,0.4],[0.4,0.5]]]).reshape(1,3,2,2)
# print('x\n',x)
# gt = torch.tensor([[[1,2],[2,0]]])
# print('gt\n',gt)
# loss,probas,true_1_hot = dice_loss(x,gt.squeeze(1))
# print ('loss\n',loss)
# print('probability\n',probas)
# print('true_1_hot\n',true_1_hot)
# tp, fp,fn = label_accuracy(probas,true_1_hot)
# print('tp\n',tp)
# print('fp\n',fp)
# print('fn\n',fn)