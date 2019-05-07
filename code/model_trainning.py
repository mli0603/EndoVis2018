import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as functional
from dice_loss import diceloss,label_accuracy
from visualization import *

# TODO: when training, turn this false
debug = False

def train(model,device,scheduler,optimizer,dice_loss,train_generator,train_dataset,writer,n_itr):
    # function to train  model for segmentation task
    # params:
        # model
        # scheduler
        # optimizer
        # dice_loss: dice loss object
        # train_generator: data generator for training set
        # train_dataset: traning dataset
        # writer: summary writer for tensorboard
        # n_iter: current iteration number, for loss plot
    scheduler.step()
    model.train()  # Set model to training mode           

    running_loss = 0.0
    
    for i_batch, batch in enumerate(train_generator):
        # read img and label
        img = batch[0]
        label = batch[1]
        
        if debug:        
            # validate if images are parsed correctly
            print(i_batch, img.shape, label.shape)
            sample_img = img[0,:,:,:]
            sample_label = label[0,:,:,:]
            sample_colorlabel = train_dataset.label_converter.label2color(sample_label.permute(1,2,0))
            imshow(sample_img.permute(1,2,0),denormalize=True)
            imshow(sample_colorlabel)

        # transfer to GPU
        img, label = img.to(device), label.to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backprop + optimize
        outputs = model(img)
        loss,_,_ = diceloss(outputs,label.long().squeeze(1))
#         loss = dice_loss.forward(outputs, label.long())
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * img.size(0)    
        writer.add_scalar('data/training_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
                
    train_loss = running_loss / len(train_dataset)
    print('Epoch Loss: {:.4f}'.format(train_loss))
    print('-' * 10)
    
    return train_loss, n_itr


def validate(model,device,dice_loss,num_class,validation_generator,validation_dataset,writer,n_itr):
    ########################### Validation #####################################
    model.eval()  # Set model to validation mode   
    validation_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, batch in enumerate(validation_generator):
        # read img and label
        img = batch[0]
        label = batch[1]

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
        outputs = model(img)
        # get loss
        loss, probas, true_1_hot = diceloss(outputs,label.long().squeeze(1))

        # statistics
        validation_loss += loss.item() * img.size(0)
        writer.add_scalar('data/validation_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn
            
    validation_loss = validation_loss / len(validation_dataset)
    print('Vaildation Loss: {:.4f}'.format(validation_loss))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, Flase Neg {}'.format(i_class, tp_val,fp_val,fn_val))
    print('-' * 10)
    
    return validation_loss, tp, fp, fn, n_itr


def test(model,device,dice_loss,num_class,test_generator,test_dataset,writer):
    ########################### Test #####################################
    model.eval()  # Set model to validation mode   
    test_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, batch in enumerate(test_generator):
        # read in images and masks
        img = batch[0]
        label = batch[1]
        
        # transfer to GPU
        img, label = img.to(device), label.to(device)

        # forward
        outputs = model(img)
        # get loss
        loss, probas, true_1_hot = diceloss(outputs,label.long().squeeze(1))

        # statistics
        test_loss += loss.item() * img.size(0)        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn 
                    
        if debug:
            break
            
    dice_score = (2*tp + 1e-7)/ (2*tp+fp+fn+1e-7)
    dice_score = dice_score.mean()
    print('Dice Score: {:.4f}'.format(dice_score.item()))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, Flase Neg {}'.format(i_class, tp_val,fp_val,fn_val))
    print('-' * 10)
    
    # visualize current prediction
    sample = test_dataset.__getitem__(0)
    img = sample[0]*0.5+0.5
    label = sample[1]
    tmp_img = sample[0].reshape(1,3,256,320)
    pred = functional.softmax(model(tmp_img.cuda()), dim=1)
    pred_label = torch.max(pred,dim=1)[1]
    pred_label = pred_label.type(label.type())
    # to plot
    tp_img = np.array(img)
    tp_label = test_dataset.label_converter.label2color(label.permute(1,2,0)).transpose(2,0,1)
    tp_pred = test_dataset.label_converter.label2color(pred_label.permute(1,2,0)).transpose(2,0,1)

    writer.add_image('Test Input', tp_img, 0)
    writer.add_image('Test Label', tp_label, 0)
    writer.add_image('Test Prediction', tp_pred, 0)
    
    return dice_score