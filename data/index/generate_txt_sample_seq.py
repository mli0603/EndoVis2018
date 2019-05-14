import random
import numpy as np
import os

seq_id = np.linspace(1,16,num=16)
seq_id = np.delete(seq_id,7).astype(int) # sequence 8 is missing
seq_num = 15
# print("List of sequence numbers")
# print(seq_id)

frame_num = 149
frame_id = range(frame_num)
# print("List of frame ids")
# print(frame_id)

total_num = seq_num*frame_num
print("Total number of frames")
print(total_num)

# open file
f_train = open("train_data.txt","w+")
f_train.write("Seq# Frame#\n")
f_validate = open("validation_data.txt","w+")
f_validate.write("Seq# Frame#\n")

for i, seq in enumerate(seq_id):
	if (i+1)%5 == 0:
		print("Seq {} in validation".format(seq))
		for frame in frame_id:
			f_validate.write(" %3d,   %03d \n" % (seq,frame))
	else:
		print("Seq {} in train".format(seq))
		for frame in frame_id:
			f_train.write(" %3d,   %03d \n" % (seq,frame))


f_train.close()
print("Train data successfully generated")
f_validate.close()
print("Validation data successfully generated")