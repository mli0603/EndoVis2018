# script to generate text file for training, validation and testing test dataset 
import random
import numpy as np
import os

seq_id = np.linspace(1,4,num=4)
seq_num = 4	
# print("List of sequence numbers")
# print(seq_id)

frame_num = 249
frame_id = range(frame_num)
# print("List of frame ids")
# print(frame_id)

total_num = seq_num*frame_num
print("Total number of frames")
print(total_num)

f = open("test_data_final.txt","w+")

f.write("Seq# Frame#\n")
for idx, seq in enumerate(seq_id):
	for frame in frame_id:
		f.write(" %3d,   %03d \n" % (seq,frame))

f.close()
print("Test data successfully generated")