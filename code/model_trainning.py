import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as functional
from dice_loss import diceloss,label_accuracy

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
        
#         # validate if images are parsed correctly
#         print(i_batch, img.shape, label.shape)
#         sample_img = img[0,:,:,:]
#         sample_label = label[0,:,:,:]
#         sample_colorlabel = train_dataset.label_converter.label2color(sample_label.permute(1,2,0))
#         imshow(sample_img.permute(1,2,0),denormalize=True)
#         imshow(sample_colorlabel)
#         break

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
        
#         # TODO: remove this!
#         break
                
    train_loss = running_loss / len(train_dataset)
    print('Epoch Loss: {:.4f}'.format(train_loss))
    print('-' * 10)
    
    return train_loss


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

#         # validate if images are parsed correctly
#         print(i_batch, img.shape, label.shape)
#         sample_img = img[0,:,:,:]
#         sample_label = label[0,:,:,:]
#         sample_colorlabel = train_dataset.label_converter.label2color(sample_label.permute(1,2,0))
#         imshow(sample_img.permute(1,2,0),denormalize=True)
#         imshow(sample_colorlabel)
#         break
        
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
        
#         # TODO: remove this!
#         break
            
    validation_loss = validation_loss / len(validation_dataset)
    print('Vaildation Loss: {:.4f}'.format(validation_loss))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, Flase Neg {}'.format(i_class, tp_val,fp_val,fn_val))
    print('-' * 10)
    
    return validation_loss, tp, fp, fn


def test():
    ########################### Test #####################################
    model.eval()  # Set model to validation mode   
    test_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, sample_batch in enumerate(test_generator):
        # read in images and masks
        img = sample_batch['image']
        mask = sample_batch['mask']
        
        # transfer to GPU
        img, mask = img.to(device), mask.to(device)

        # forward
        outputs = model(img)
        # get loss
        loss, probas, true_1_hot = dice_loss(outputs,mask.long().squeeze(1))

        # statistics
        test_loss += loss.item() * img.size(0)        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn
        
        # visualize current prediction
        if i_batch == 3:
            imshow(img.cpu(),denormalize=True)
            imshow(class2mask(mask).cpu())
            pred = functional.softmax(outputs, dim=1)
            pred_label = torch.max(pred,dim=1)[1]
            pred_mask = class2mask(pred_label)
            pred_mask = torch.reshape(pred_mask,(pred_mask.shape[0],1,pred_mask.shape[1],pred_mask.shape[2]))
            imshow(pred_mask.cpu())
            
    dice = (2*tp + 1e-7)/ (2*tp+fp+fn+1e-7)
    dice = dice.mean()
    print('Dice Score: {:.4f}'.format(dice.item()))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, Flase Neg {}'.format(i_class, tp_val,fp_val,fn_val))
    print('-' * 10)
    
    return dice