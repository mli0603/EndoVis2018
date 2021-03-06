import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as functional
from dice_loss import * 
from visualization import *
import random
import copy


# TODO: when training, turn this false
debug = False

def train(model,device,scheduler,optimizer,dice_loss,num_class,train_generator,train_dataset,writer,n_itr):
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
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, batch in enumerate(train_generator):
        # read img and label
        img = batch['img']
        label = batch['label']
        
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
        loss,probas,true_1_hot = dice_loss.forward(outputs, label.long())
        loss.backward()
        optimizer.step()
        loss.detach()

        # statistics
        running_loss += loss.item() * img.size(0)    
        writer.add_scalar('data/training_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn
        
        if debug:
            break
                
    train_loss = running_loss / len(train_dataset)
    print('Training Loss: {:.4f}'.format(train_loss))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, False Neg {}, Num Pixel {}, Dice score {:1.2f}'.format(i_class, tp_val,fp_val,fn_val,tp_val+fn_val,(2*tp_val + 1e-7)/ (2*tp_val+fp_val+fn_val+1e-7)))
    
    return train_loss, n_itr


def validate(model,device,dice_loss,num_class,validation_generator,validation_dataset,writer,n_itr):
    ########################### Validation #####################################
    model.eval()  # Set model to validation mode   
    validation_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
   
    worst_dice = np.ones((5,1))
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
        outputs = model(img)
        # get loss
        loss, probas, true_1_hot = dice_loss.forward(outputs, label.long())
        loss.detach()

        # statistics
        validation_loss += loss.item() * img.size(0)
        writer.add_scalar('data/validation_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn
        curr_dice = ((2*curr_tp + 1e-7)/ (2*curr_tp+curr_fp+curr_fn+1e-7)).mean()
        
        # find best and worst
        if np.max(worst_dice) > curr_dice:
            idx = np.argmax(worst_dice)
            worst_batch[idx] = batch
            worst_dice[idx] = curr_dice
            
        if np.min(best_dice) < curr_dice:
            idx = np.argmin(best_dice)
            best_batch[idx] = batch
            best_dice[idx] = curr_dice
        
        if debug:
            break
            
    validation_loss = validation_loss / len(validation_dataset)
    print('Vaildation Loss: {:.4f}'.format(validation_loss))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, False Neg {}, Num Pixel {}, Dice score {:1.2f}'.format(i_class, tp_val,fp_val,fn_val,tp_val+fn_val,(2*tp_val + 1e-7)/ (2*tp_val+fp_val+fn_val+1e-7)))
    print('-' * 10)
    
    return validation_loss, tp, fp, fn, n_itr,[worst_batch,worst_dice],[best_batch,best_dice]


def test(model,device,dice_loss,num_class,test_generator,test_dataset,writer):
    ########################### Test #####################################
    model.eval()  # Set model to validation mode   
    test_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, batch in enumerate(test_generator):
        # read in images and masks
        img = batch['img']
        label = batch['label']
        
        # transfer to GPU
        img, label = img.to(device), label.to(device)

        # forward
        outputs = model(img)
        # get loss
        loss, probas, true_1_hot = dice_loss.forward(outputs, label.long())

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
    # sample = test_dataset.__getitem__(0)
    # img = sample[0]*0.5+0.5
    # label = sample[1]
    # tmp_img = sample[0].reshape(1,3,256,320)
    # pred = functional.softmax(model(tmp_img.cuda()), dim=1)
    # pred_label = torch.max(pred,dim=1)[1]
    # pred_label = pred_label.type(label.type())
    # # to plot
    # tp_img = np.array(img)
    # tp_label = test_dataset.label_converter.label2color(label.permute(1,2,0)).transpose(2,0,1)
    # tp_pred = test_dataset.label_converter.label2color(pred_label.permute(1,2,0)).transpose(2,0,1)

    # writer.add_image('Test Input', tp_img, 0)
    # writer.add_image('Test Label', tp_label, 0)
    # writer.add_image('Test Prediction', tp_pred, 0)
    
    return dice_score

def run_training(model,device,num_class,scheduler,optimizer,dice_loss,num_epochs,train_generator,train_dataset,validation_generator,validation_dataset,writer,model_path):
    print("Training Started!")

    # initialize best_acc for comparison
    best_acc = 0.0
    train_iter = 0
    val_iter = 0

    for epoch in range(num_epochs):
        print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")

        # train
        train_loss, train_iter = train(model,device,scheduler,optimizer,dice_loss,num_class,train_generator,train_dataset,writer,train_iter)

        # validate
        with torch.no_grad():
            validation_loss, tp, fp, fn, val_iter, worst, best = validate(model,device,dice_loss,num_class,validation_generator,validation_dataset,writer,val_iter)
            epoch_acc = (2*tp + 1e-7)/ (2*tp+fp+fn+1e-7)
            epoch_acc = epoch_acc.mean()
    
            # loss
            writer.add_scalar('data/Training Loss (per epoch)',train_loss,epoch)
            writer.add_scalar('data/Validation Loss (per epoch)',validation_loss,epoch)
                
            # show best and worst 
            # print("worst performance: dice {:.2f}".format(worst[1]))
            # batch = worst[0]
            # img = batch['img']*0.5+0.5
            # label = batch['label']
            # tmp_img = batch['img']
            # pred = functional.softmax(model(tmp_img.cuda()), dim=1)
            # tmp_img.cpu()
            # pred_label = torch.max(pred,dim=1)[1]
            # pred_label = pred_label.cpu().reshape([pred_label.shape[0],1,pred_label.shape[1],pred_label.shape[2]]).type(label.type())
            # # make grid
            # tp_img = torchvision.utils.make_grid(img,nrow=3).numpy()
            # tp_label = torchvision.utils.make_grid(label,nrow=3)[0,:,:]
            # tp_label = tp_label.reshape([1,tp_label.shape[0],tp_label.shape[1]])
            # tp_label = train_dataset.label_converter.label2color(tp_label.permute(1,2,0)).transpose(2,0,1)
            # tp_pred = torchvision.utils.make_grid(pred_label,nrow=3)[0,:,:]
            # tp_pred = tp_pred.reshape([1,tp_pred.shape[0],tp_pred.shape[1]])
            # tp_pred = train_dataset.label_converter.label2color(tp_pred.permute(1,2,0)).transpose(2,0,1)

            # writer.add_image('Worst Input', tp_img, epoch)
            # writer.add_image('Worst Label', tp_label, epoch)
            # writer.add_image('Worst Prediction', tp_pred, epoch)            

            # print("best performance: dice {:.2f}".format(best[1]))
            # batch = best[0]
            # img = batch['img']*0.5+0.5
            # label = batch['label']
            # tmp_img = batch['img']
            # pred = functional.softmax(model(tmp_img.cuda()), dim=1)
            # tmp_img.cpu()
            # pred_label = torch.max(pred,dim=1)[1]
            # pred_label = pred_label.cpu().reshape([pred_label.shape[0],1,pred_label.shape[1],pred_label.shape[2]]).type(label.type())
            # # make grid
            # tp_img = torchvision.utils.make_grid(img,nrow=3).numpy()
            # tp_label = torchvision.utils.make_grid(label,nrow=3)[0,:,:]
            # tp_label = tp_label.reshape([1,tp_label.shape[0],tp_label.shape[1]])
            # tp_label = train_dataset.label_converter.label2color(tp_label.permute(1,2,0)).transpose(2,0,1)
            # tp_pred = torchvision.utils.make_grid(pred_label,nrow=3)[0,:,:]
            # tp_pred = tp_pred.reshape([1,tp_pred.shape[0],tp_pred.shape[1]])
            # tp_pred = train_dataset.label_converter.label2color(tp_pred.permute(1,2,0)).transpose(2,0,1)

            # writer.add_image('best Input', tp_img, epoch)
            # writer.add_image('best Label', tp_label, epoch)
            # writer.add_image('best Prediction', tp_pred, epoch)         

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Dice Score: {:.4f}'.format(best_acc.item()))

                if epoch > 10:
                	torch.save(model.state_dict(), model_path)
                
            print('-' * 10)
            
    return best_model_wts, best_acc
    
    
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
    
