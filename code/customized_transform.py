from albumentations import *
from albumentations.pytorch import *
from visualization import *
import numpy as np
import cv2
from dataset import *
import random
import albumentations.augmentations.functional as F

    
class RandomSpotlight(ImageOnlyTransform):
    """Simulates spot light
    Args:
        flare_roi (float, float, float, float): region of the image where flare will
                                                    appear (x_min, y_min, x_max, y_max)
        src_radius (int)
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self,
                 flare_roi=(0, 0, 1, 1),
                 radius_limit = (30,50),
                 alpha_limit = (0.3,0.7),
                 color_limit = (0.3,0.7),
                 always_apply=False,
                 p=0.5):
        super(RandomSpotlight, self).__init__(always_apply, p)

        (flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y) = flare_roi
        (self.src_radius_lower, self.src_radius_upper) = radius_limit
        (self.src_alpha_lower, self.src_alpha_upper) = alpha_limit

        assert 0 <= flare_center_lower_x < flare_center_upper_x <= 1
        assert 0 <= flare_center_lower_y < flare_center_upper_y <= 1

        self.flare_center_lower_x = flare_center_lower_x
        self.flare_center_upper_x = flare_center_upper_x

        self.flare_center_lower_y = flare_center_lower_y
        self.flare_center_upper_y = flare_center_upper_y

    def apply(self,
              image,
              flare_center_x=0.5,
              flare_center_y=0.5,
              radius = 50,
              alpha = 0.5,
              **params):
        
        image = np.array(image)
        
        overlay = np.zeros_like(image)
        output = image.copy()
        
        if radius % 2==0:
            kernel = radius + 1
        else:
            kernel = radius

        overlay = cv2.circle(overlay,(flare_center_x,flare_center_y),radius,(255,255,255),-1)
        blur = cv2.GaussianBlur(overlay,(kernel,kernel),int(kernel/2))
    
        output = cv2.addWeighted(blur,alpha,output,1,0,-1)
        
        return output

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]

        flare_center_x = random.uniform(self.flare_center_lower_x, self.flare_center_upper_x)
        flare_center_y = random.uniform(self.flare_center_lower_y, self.flare_center_upper_y)

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)
        
        radius = random.uniform(self.src_radius_lower, self.src_radius_upper)
        radius = int(radius)
        
        alpha = random.uniform(self.src_alpha_lower, self.src_alpha_upper)
                
        return {'radius': radius,
                'alpha': alpha,
                'flare_center_x': flare_center_x,
                'flare_center_y': flare_center_y}
    

class ThreadHueSaturationValue(HueSaturationValue):
    """Randomly change hue, saturation and value of the input image.
    Args:
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: 20.
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: 30.
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: 20.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, hue_shift_limit=50, sat_shift_limit=30, val_shift_limit=20, always_apply=False, thread_idx = 6, p=0.5):
        super(ThreadHueSaturationValue, self).__init__(always_apply, p)
        print("init")
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)
        self.thread_idx = 6

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, label = None, **params):
        print("apply")
        if label is not None:
            print('thread aug')
            image = np.array(image)
        
            output = image.copy()
            hsv = image.copy()
            hsv = F.shift_hsv(hsv, hue_shift, sat_shift, val_shift)
        
            thread = np.where(label==self.thread_idx)
            output[thread] = hsv[thread]
            
            return output
        else:
            print('thread aug failed')
            return image

    def get_params(self):
        print("get param")
        return {'hue_shift': random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
                'sat_shift': random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
                'val_shift': random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])}
    
    def update_params(self, params, **kwargs):
        print("update params")
        params.update({'label': kwargs['label']})
        return params


    
if __name__ == "__main__":
    img_path = "../data/"+"images/seq_"+"1"+"/left_frames/frame"+"001"+".png"
    img = Image.open(img_path)
    img = img.resize((320, 256))
    imshow(img)
    
    image = np.array(img)
    overlay = np.zeros_like(image)
    output = image.copy()
    
    radius = 50
    if radius % 2==0:
        kernel = radius + 1
    else:
        kernel = radius
        
    overlay = cv2.circle(overlay,(150,150),radius,(255,255,255),-1)
    blur = cv2.GaussianBlur(overlay,(kernel,kernel),int(kernel/2))
    
    output = cv2.addWeighted(blur,0.5,output,1,0,-1)
    imshow(output)