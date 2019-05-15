import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as functional
from dice_loss import * 
from visualization import *
import random
import copy
from PIL import Image


# TODO: when training, turn this false
debug = False

def inference(model,device,dice_loss,num_class,num_superclasses,super2sub,test_generator,test_dataset,writer,save_path):
    ########################### Test #####################################
    model.eval()  # Set model to validation mode   
    test_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, batch in enumerate(test_generator):
        # read in images and masks
        img = batch['img']
        idx = batch['idx']
        
        # transfer to GPU
        img = img.to(device)

        # forward
        outputs = model(img)

        # label and plot and save
        superclass_scores = outputs[0]
        class_score = outputs[1]
        final_class_score = torch.zeros_like(class_score)
        for i in range(num_superclasses):
            for j in super2sub[i]:
                final_class_score[:,j,:,:] = class_score[:,j,:,:]*superclass_scores[:,i,:,:]

        pred = functional.softmax(final_class_score.cpu(), dim=1)
        img.cpu()

        pred_label = torch.max(pred,dim=1)[1]
        pred_label = pred_label.cpu()

        sample_img = (img[0,:,:,:].cpu().permute(1,2,0)*0.5+0.5)*255
        sample_predcolorlabel = test_dataset.label_converter.label2color(pred_label.permute(1,2,0))

        # print(sample_img.shape)
        # print(sample_predcolorlabel.shape)

        output_img = np.concatenate((np.uint8(sample_img),np.uint8(sample_predcolorlabel)),axis=1)
        output_img = Image.fromarray(output_img)

        output_img.save(save_path+str(idx.item())+".png","PNG")

        # imshow(sample_img.cpu().permute(1,2,0),denormalize=True)
        # imshow(sample_predcolorlabel)

                    
        if debug:
            break
            
    print('inference finished:')
    print('-' * 10)
    
    return    
    
def validate_and_compare(model1,model2,device,dice_loss1,dice_loss2,num_class,validation_generator,validation_dataset):
    ########################### Validation #####################################
    model1.eval()  # Set model to validation mode   
    model2.eval()  # Set model to validation mode   
    
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    worst_dice = np.zeros((5,1))
    best_dice = np.zeros((5,1))
    
    worst_batch = [0,0,0,0,0]
    best_batch = [0,0,0,0,0]
    
    for i_batch, batch in enumerate(validation_generator):
        # read img and label
        img = batch['img']
        label = batch['label']
        indx = batch['indx']
        
        if debug:
            # validate if images are parsed correctly
            print(i_batch, img.shape, label.shape)
            sample_img = img[0,:,:,:]
            sample_label = label[0,:,:,:]
            sample_colorlabel = validation_dataset.label_converter.label2color(sample_label.permute(1,2,0))
            imshow(sample_img.permute(1,2,0),denormalize=True)
            imshow(sample_colorlabel)
        
        # transfer to GPU
        img, label = img.to(device), label.to(device)

        # forward
        outputs1 = model1(img)
        outputs2 = model2(img)
        # get loss
        _, probas1, true_1_hot = dice_loss1.forward(outputs1, label.long())
        _, probas2, true_1_hot = dice_loss2.forward(outputs2, label.long())
        
        curr_tp1, curr_fp1, curr_fn1 = label_accuracy(probas1.cpu(),true_1_hot.cpu())
        curr_tp2, curr_fp2, curr_fn2 = label_accuracy(probas2.cpu(),true_1_hot.cpu())
        
        curr_dice1 = ((2*curr_tp1 + 1e-7)/ (2*curr_tp1+curr_fp1+curr_fn1+1e-7)).mean()
        curr_dice2 = ((2*curr_tp2 + 1e-7)/ (2*curr_tp2+curr_fp2+curr_fn2+1e-7)).mean()
        
        diff_dice = curr_dice2-curr_dice1
        
        # find best and worst
        if np.max(worst_dice) > diff_dice:
            idx = np.argmax(worst_dice)
            worst_batch[idx] = batch
            worst_dice[idx] = diff_dice
            
        if np.min(best_dice) < diff_dice:
            idx = np.argmin(best_dice)
            best_batch[idx] = batch
            best_dice[idx] = diff_dice
        
        if debug:
            break
    
    return [worst_batch,worst_dice],[best_batch,best_dice]
    
