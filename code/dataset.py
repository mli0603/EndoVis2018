import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import json

from label_conversion import LabelConverter
from dice_loss import label_accuracy
from shuffle_puzzle import Puzzle_RandomShuffle
from visualization import *
from albumentations import *
from albumentations.pytorch import *
from customized_transform import *

class MICCAIDataset(Dataset):
    def __init__(self, data_path="../data/", data_type = "train", version = "_min", transform_both=None, transform_image=None, transform_per_class=None):
        #store some input 
        self.data_path = str(data_path)
        self.data_type = str(data_type)
        self.filename = data_path+"index/"+data_type+"_data"+version+".txt"
        self.transform_both = transform_both
        self.transform_image = transform_image
        self.transform_per_class = transform_per_class
        self.data = []

        #parse the txt to store the necessary information of output
        file = open(self.filename, 'r').readlines()
        file = file[1:]
        for i in range(len(file)):
            file[i] = file[i].split(",")
            entry = {}
            entry["seq"] = file[i][0].strip()
            entry["frame"] = file[i][1].strip().zfill(3)
            self.data.append(entry)

        # save label conversion object
        self.label_converter = LabelConverter(data_path)

    def __len__(self):
        #return the length of the data numbers
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == "train" or self.data_type == "validattion":
            prefix = "images/seq_"
            label_path = self.data_path+prefix+self.data[idx]["seq"]+"/labels/frame"+self.data[idx]["frame"]+".png"
            #parse label color to label number and resize it to 320x256
            label = Image.open(label_path)
            label = label.resize((320, 256))
            label = np.array(label, dtype='int32')
            label_indx = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]
            label = self.label_converter.color2label(label_indx)
        else:
            prefix = "test/seq_"
        
        img_path = self.data_path+prefix+self.data[idx]["seq"]+"/left_frames/frame"+self.data[idx]["frame"]+".png"
        

        #get img from file and resize it to 320x256 which is what we want
        img = Image.open(img_path)
        img = img.resize((320, 256))
        img = np.array(img)
            
        # augment dataset
        if self.transform_both is not None:
            augmented = self.transform_both(image=img,mask=label)
            img = augmented['image']
            label = augmented['mask']
            
        if self.transform_per_class is not None:    
            print('per class')
            augmented = self.transform_per_class(image=img,label=label)
            img = augmented['image']
            
        if self.transform_image is not None:
            augmented = self.transform_image(image=img)
            img = augmented['image']
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.data_type == "train" or self.data_type == "validattion":
            label = torch.from_numpy(label).reshape([1,label.shape[0],label.shape[1]])
            sample = {'img':img,'label':label,'indx':idx}
        else:
            sample = {'img':img,'idx':idx}
        
        return sample

class Transformation_PretrainDataset(MICCAIDataset):
    def __init__(self, data_path="../data/", data_type = "train", transform=None):
        #store some input 
        super(Transformation_PretrainDataset, self).__init__(data_path, data_type, transform)
        self.randomShiftScaleRotate = transforms.ShiftScaleRotate(prob=1.0)

    def __getitem__(self, idx):
        img_path = self.data_path+"images/seq_"+self.data[idx]["seq"]+"/left_frames/frame"+self.data[idx]["frame"]+".png"
        label_path = self.data_path+"images/seq_"+self.data[idx]["seq"]+"/right_frames/frame"+self.data[idx]["frame"]+".png"
        #get img from file and resize it to 320x256 which is what we want
        img = Image.open(img_path)
        img = img.resize((320, 256))
        # change to 3 channels
        img = np.array(img)
        label = Image.open(label_path)
        label = label.resize((320, 256))
        label = np.array(label)
        
        [label,_] = self.randomShiftScaleRotate(img,mask=None)      
        
        # apply normalization
        norm = transforms.Normalize()
        img = norm(img)
        label = norm(label)
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)
        
        sample = {'img':img,'label':label,'idx':idx}
        
        return sample

class Colorize_PretrainDataset(MICCAIDataset):
    def __init__(self, data_path="../data/", data_type = "train", transform=None):
        #store some input 
        super(Colorize_PretrainDataset, self).__init__(data_path, data_type, transform)

    def __getitem__(self, idx):
        img_path = self.data_path+"images/seq_"+self.data[idx]["seq"]+"/left_frames/frame"+self.data[idx]["frame"]+".png"
        img = Image.open(img_path).convert('L')
        img = img.resize((320, 256))
        img = np.array(img)
#         print(img.shape)
        label = Image.open(img_path)
        label = label.resize((320, 256))
        label = np.array(label)
            
        img = torch.from_numpy(img)
        label = torch.from_numpy(label).permute(2, 0, 1)
        
        sample = {'img':img,'label':label,'indx':idx}
        return sample

class Shuffle_PretrainDataset(MICCAIDataset):
    def __init__(self, data_path="../data/", data_type = "train", transform=None, n=30,seed =1):
        self.data_path = data_path
        self.data_type = data_type
        self.transform = transform
        self.filename = data_path+"index/"+data_type+"_data.txt"
        self.n = n
        self.seed = seed
        self.data = []

        #parse the txt to store the necessary information of output
        file = open(self.filename, 'r').readlines()
        file = file[1:]
        for i in range(len(file)):
            file[i] = file[i].split(",")
            entry = {}
            entry["seq"] = file[i][0].strip()
            entry["frame"] = file[i][1].strip().zfill(3)
            self.data.append(entry)
        #store some input 
        #super(Shuffle_PretrainDataset, self).__init__(data_path, data_type, transform,n)
    def __len__(self):
        #return the length of the data numbers
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data_path+"images/seq_"+self.data[idx]["seq"]+"/left_frames/frame"+self.data[idx]["frame"]+".png"
        img = Puzzle_RandomShuffle(img_path, self.n, self.seed)
        img = img.resize((320, 256))
        img = np.array(img)

        label = Image.open(img_path)
        label = label.resize((320, 256))
        label = np.array(label)
        
        # augment dataset
        # if self.transform is not None:
        #     img,label = transforms.augment(img,label) 
        #     # apply totensor and normalization only to img
        #     norm = transforms.Normalize()
        #     img = norm(img)
        #pil2tensor = transforms.ToTensor()    
        #img = pil2tensor(img)
        
        # apply normalization
        norm = transforms.Normalize()
        img = norm(img)
        label = norm(label)
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)
        
        sample = {'img':img,'label':label,'indx':idx}
        
        return sample


if __name__ == "__main__":
    label_converter = LabelConverter()
    
    train_both_aug = Compose([
        Cutout(num_holes=8,p=0.5),
        OneOf([
            ShiftScaleRotate(rotate_limit=15,p=0.5),
            HorizontalFlip(p=0.8),
        ])
    ])
    train_image_aug = Compose([
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,p=0.7),
            RandomGamma(gamma_limit=(50,200),p=0.7),        
            HueSaturationValue(p=0.7),            
        ]),
        MotionBlur(blur_limit=7,p=0.7),
        RandomSpotlight(p=0.7),
        Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),p=1),
    ])
    
    train_per_class_aug = Compose([
        ThreadHueSaturationValue(hue_shift_limit= (-20,20), sat_shift_limit = (-30,30), val_shift_limit=(-20,20), always_apply=False,p=1.0),
        Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),p=1),
    ])
    
    val_image_aug = Compose([
        Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),p=1),
    ])
    
  
    train_dataset=MICCAIDataset(data_type="train",                                transform_both=None,transform_image=None,transform_per_class=train_per_class_aug)
    train_generator = DataLoader(train_dataset,shuffle=False,batch_size=1,num_workers=1)
    
    for i_batch, sample_batch in enumerate(train_generator):
        img = sample_batch['img']
        label = sample_batch['label']
        
        print(img.shape)
        print(label.shape)
        
        imshow(img[0,:,:,:].permute(1,2,0),denormalize=True)
        tmp = label_converter.label2color(label[0,:,:,:].permute(1,2,0))
        imshow(tmp,denormalize=False)
        break