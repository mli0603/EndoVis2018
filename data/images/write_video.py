import cv2
import numpy as np
import glob, os
from natsort import natsorted, ns

frame_size = (320*2,256)
out = cv2.VideoWriter('seq.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, frame_size)

for root, dirs, files in os.walk("C:/Users/Maxwell/Downloads/output_seq"):
    for file in natsorted(files):
        if file.endswith(".png"):
             frame = cv2.imread(os.path.join(root, file))

             out.write(frame)

out.release()