import numpy as np
import json

class LabelConverter():
    def __init__(self,data_path):
        #parse labels.json
        f = open(data_path+"labels.json").read()
        labels = json.loads(f)
        self.color2label = np.zeros(256**3)
        self.label2name = []
        self.class_num = len(labels)

        for i in range(len(labels)):
            color = labels[i]["color"]
            self.color2label[(color[0]*256+color[1])*256+color[2]] = i
            self.label2name.append(labels[i]["name"])

    def label2color(self,label):
        # TODO
        return

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