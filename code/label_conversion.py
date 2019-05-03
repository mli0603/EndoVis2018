import numpy as np
import json
import matplotlib.pyplot as plt

class LabelConverter():
    def __init__(self,data_path):
        #parse labels.json
        f = open(data_path+"labels.json").read()
        labels = json.loads(f)
        self._color2label = np.zeros(256**3)
        self._label2name = []
        self._label2color = []
        self.class_num = len(labels)

        for i in range(len(labels)):
            color = labels[i]["color"]
            self._color2label[(color[0]*256+color[1])*256+color[2]] = i
            self._label2name.append(labels[i]["name"])
            self._label2color.append(color)
        self._label2color = np.array(self._label2color)
    def label2color(self,label):
        #convert labels to RGB colors for visulization
        #input:
            #label: label image (w*h)
        #output:
            #image: colored image (w*h*3)
        img = self._label2color[label].astype(np.uint8)
        return img

    def label2superlabel(self,label):
        # merge different labels into one super for pre-training
        # input: 
            # label: label image
        # output:
            # superlabel: image, a super class of the labels
            # 0: background-tissue (0), kidney-parenchyma (4), covered-kidney (5), thread(6), small-intestine (10)
            # 1: instrument-shaft (1), instrument-clasper (2), instrument-wrist (3), clamps (7), suturing-needle (8), 
            # suction-instrument(9), ultrasound-probe (11)
            
        superlabel = label
        
        for currlabel in range (0,11):
            if currlabel in set([0,4,5,6,10]):
                superlabel[superlabel==currlabel] = 0
            elif currlabel in set([1,2,3,7,8,9,11]):
                superlabel[superlabel==currlabel] = 1
                
        return superlabel

if __name__=="__main__":
    label_converter = LabelConverter("../data/")
    label = np.ones(3)*6
    print(label_converter.label2superlabel(label))
    label = np.ones((320,256)).astype(np.uint32)
    for i in range(10):
        label[i*32:(i+1)*32,i*25:(i+1)*25] = i
    img = label_converter.label2color(label)
    print(img)
    plt.imshow(img)
    plt.show()
