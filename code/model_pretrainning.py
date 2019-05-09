import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as functional
from visualization import *
import random
import copy

# TODO: when training, turn this false
debug = False

def pretrain(model,device,scheduler,optimizer,criterion,train_generator,train_dataset,writer,n_itr):
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
            imshow(sample_img.permute(1,2,0),denormalize=True)
            imshow(sample_label.permute(1,2,0),denormalize=True)

        # transfer to GPU
        img, label = img.to(device), label.to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backprop + optimize
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * img.size(0)    
        writer.add_scalar('data/pre_training_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        if debug:
            break
                
    train_loss = running_loss / len(train_dataset)
    print('Epoch Loss: {:.4f}'.format(train_loss))
    print('-' * 10)
    
    return train_loss, n_itr


def prevalidate(model,device,criterion,validation_generator,validation_dataset,writer,n_itr):
    ########################### Validation #####################################
    model.eval()  # Set model to validation mode   
    validation_loss = 0.0
    
    for i_batch, batch in enumerate(validation_generator):
        # read img and label
        img = batch[0]
        label = batch[1]

        if debug:
            # validate if images are parsed correctly
            print(i_batch, img.shape, label.shape)
            sample_img = img[0,:,:,:]
            sample_label = label[0,:,:,:]
            imshow(sample_img.permute(1,2,0),denormalize=True)
            imshow(sample_label.permute(1,2,0),denormalize=True)
        
        # transfer to GPU
        img, label = img.to(device), label.to(device)

        # forward
        outputs = model(img)
        # get loss
        loss = criterion(outputs, label)

        # statistics
        validation_loss += loss.item() * img.size(0)
        writer.add_scalar('data/pre_validation_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        if debug:
            break
            
    validation_loss = validation_loss / len(validation_dataset)
    print('Vaildation Loss: {:.4f}'.format(validation_loss))
    
    return validation_loss, n_itr

def run_pretraining(model,device,scheduler,optimizer,criterion,num_epochs,pretrain_generator,pretrain_dataset,prevalidation_generator,prevalidation_dataset,writer):
    print("Pre-Training Started!")

    # initialize best_acc for comparison
    best_loss = 1000.0
    train_iter = 0
    val_iter = 0

    for epoch in range(num_epochs):
        print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")

        # train
        train_loss, train_iter = pretrain(model,device,scheduler,optimizer,criterion,pretrain_generator,pretrain_dataset,writer,train_iter)

        # validate
        with torch.no_grad():
            validation_loss, val_iter = prevalidate(model,device,criterion,prevalidation_generator,prevalidation_dataset,writer,val_iter)

            # loss
            writer.add_scalar('data/Pre-Training Loss (per epoch)',train_loss,epoch)
            writer.add_scalar('data/Pre-Validation Loss (per epoch)',validation_loss,epoch)

            # randomly show one validation image 
            sample = prevalidation_dataset.__getitem__(random.randint(0,len(prevalidation_dataset)-1))
            img = sample[0]*0.5+0.5
            label = sample[1]*0.5+0.5
            tmp_img = sample[0].reshape(1,3,256,320)
            pred = model(tmp_img.cuda())
            pred = pred*0.5+0.5
            # to plot
            tp_img = np.array(img)
            tp_label = np.array(label)
            tp_pred = np.array(pred.cpu().squeeze())
            
            if debug:
                print(tp_label.shape)
                print(tp_pred.shape)

            writer.add_image('Pre-Input', tp_img, epoch)
            writer.add_image('Pre-Label', tp_label, epoch)
            writer.add_image('Pre-Prediction', tp_pred, epoch)

            # deep copy the model
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    
    return best_model_wts