# script to generate text file for training, validation and testing test dataset 
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
total_id = range(total_num)
print("Total number of frames")
print(total_num)

# split test:(training+validation) = 2:8
test_percentage = 0.2
train_percentage = 0.8
validation_percentage = 0.2

# check if test file exists, this makes sure we don't alternate test images
test_num = int(test_percentage*total_num)
if not os.path.isfile('test_data.txt'):
	# generate test index
	test_id = random.sample(range(total_num), test_num)
	test_id = np.array(test_id)

	f = open("test_data.txt","w+")
	f.write("Seq# Frame#\n")
	for index in np.nditer(test_id):
		# print ("index ")
		# print (index)
		# print("seq id")
		seq_tmp = seq_id[index / 149]
		# print(seq_tmp)
		# print("frame id")
		frame_tmp = index - 149 * (index / 149) # zero-indexed
		# print(frame_tmp)
		f.write(" %3d,   %03d \n" % (seq_tmp,frame_tmp))
	f.close()
	print("Test data successfully generated")
else:
	f = open("test_data.txt","r+")
	f1 =f.readlines()
	test_id =[]
	for line in f1:
		line = line.split(',')
		# the first line has length of 1 because there is no comma
		if len(line) > 1:
			seq_tmp=int(line[0])
			frame_tmp=int(line[1])
			# print(seq_tmp,frame_tmp)
			test_id.append(seq_tmp*149+frame_tmp)
			# print(test_id)
	test_id = np.array(test_id)
	# print(contents)
	f.close()
	print("Test data already exists, don't generate test samples")

print("Number of test data")
print(test_num)

# generate training+validation index
validation_num = int(validation_percentage*(total_num-test_num))
train_num = total_num - test_num - validation_num
remaining_id = np.delete(total_id,test_id)
subsample_for_validation = random.sample(range(total_num-test_num), validation_num)
subsample_for_validation = np.array(subsample_for_validation)
train_id = np.delete(remaining_id,subsample_for_validation)
validation_id = remaining_id[subsample_for_validation]
print('Number of validation data')
print(validation_num)
print('Number of train data')
print(train_num)

# generate list of train images
f = open("train_data.txt","w+")
f.write("Seq# Frame#\n")
for index in np.nditer(train_id):
	# print ("index ")
	# print (index)
	# print("seq id")
	seq_tmp = seq_id[index / 149]
	# print(seq_tmp)
	# print("frame id")
	frame_tmp = index - 149 * (index / 149) # zero-indexed
	# print(frame_tmp)
	f.write(" %3d,   %03d \n" % (seq_tmp,frame_tmp))
f.close()
print("Train data successfully generated")

# generate list of validation images
f = open("validation_data.txt","w+")
f.write("Seq# Frame#\n")
for index in np.nditer(validation_id):
	# print ("index ")
	# print (index)
	# print("seq id")
	seq_tmp = seq_id[index / 149]
	# print(seq_tmp)
	# print("frame id")
	frame_tmp = index - 149 * (index / 149) # zero-indexed
	# print(frame_tmp)
	f.write(" %3d,   %03d \n" % (seq_tmp,frame_tmp))
f.close()
print("Validation data successfully generated")