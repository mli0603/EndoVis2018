# script to generate text file for training, validation and testing test dataset 
import random
import numpy as np

f = open("train_data.txt","r")
index = f.readlines()
title = index[0]
index = index[386:395]
f.close()
f = open("class8train_data.txt","w")
f.write(title)
for i in range(len(index)):
	f.write(index[i])
