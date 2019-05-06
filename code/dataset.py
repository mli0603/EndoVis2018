import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from  PIL import Image
import json
import transforms

from label_conversion import LabelConverter
from visualization import *

class MICCAIDataset(Dataset):
    def __init__(self, data_path="../data/", data_type = "train", transform=None):
        #store some input 
        self.data_path = str(data_path)
        self.data_type = str(data_type)
        self.filename = data_path+"index/"+data_type+"_data.txt"
        self.transform = transform
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
        img_path = self.data_path+"images/seq_"+self.data[idx]["seq"]+"/left_frames/frame"+self.data[idx]["frame"]+".png"
        label_path = self.data_path+"images/seq_"+self.data[idx]["seq"]+"/labels/frame"+self.data[idx]["frame"]+".png"

        #get img from file and resize it to 320x256 which is what we want
        img = Image.open(img_path)
        img = img.resize((320, 256))
        img = np.array(img)
#         print(img.shape)

        #parse label color to label number and resize it to 320x256
        label = Image.open(label_path)
#         plt.imshow(label)
#         plt.show()
        label = label.resize((320, 256))
        label = np.array(label, dtype='int32')
        idx = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]
        label = self.label_converter.color2label(idx)
        
        # augment dataset
        if self.transform is not None:
            img,label = transforms.augment(img,label) 
        
        # apply totensor and normalization only to img
        norm = transforms.Normalize()
        img = norm(img)
            
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.from_numpy(label).reshape([1,label.shape[0],label.shape[1]])
        
        return img,label

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
        img = np.array(img)
        label = Image.open(label_path)
        label = label.resize((320, 256))
        label = np.array(label)
        
        [label,_] = self.randomShiftScaleRotate(img,mask=None)      
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)
        return img,label

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
        return img,label



if __name__ == "__main__":
    dataset=MICCAIDataset(transform=transforms)
    label_converter = LabelConverter()
    idx = 0
    imshow(dataset[idx][0].permute(1,2,0),denormalize=True)
    tmp = label_converter.label2color(dataset[idx][1].permute(1,2,0))
    imshow(tmp)
    
    dataset = Colorize_PretrainDataset()
    idx = 0
    plt.imshow(dataset[idx][1].permute(1,2,0))
    plt.show()
    plt.imshow(dataset[idx][0], cmap="gray")
    plt.show()

    dataset = Transformation_PretrainDataset()
    idx = 0
    plt.imshow(dataset[idx][0].permute(1,2,0))
    plt.show()
    plt.imshow(dataset[idx][1].permute(1,2,0))
    plt.show()