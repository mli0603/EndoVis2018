from visdom import Visdom
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# helper function to show images from a batch of tensor
def imshow_batch(inp, denormalize=False):
    """Imshow for batch of Tensor."""
    
    # Make a grid from batch
    inp = torchvision.utils.make_grid(inp)

    # convert and display
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    if denormalize:
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
# helper function to show images
def imshow(img, denormalize=False):   
    # param:
        # img: tensor or np array (W,H,C)
    img = np.array(img)
    
    mean = 0.5
    std = 0.5
    if denormalize:
        img = std * img + mean
#         print(np.max(img))
#         print(np.min(img))
        img = np.clip(img, 0, 1)
        

    plt.imshow(img)
    plt.show()

# visdom bridge to visualize loss and images
class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = Visdom(env=self.env_name)
        self.loss_win = None
        self.seg_win = 0

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            loss.reshape(1,2),
            np.array([step,step]).reshape(1,2),
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Loss (mean loss per 1 epoch)',
                legend=['training', 'validation']
            )
        )
        
    def plot_image(self, img, mask, pred):
        # param:
            # img: tensor or np array, (C,W,H)
        self.vis.image(
            img,
            win=self.seg_win,
            opts=dict(title='Sample Segmentation Input', caption='Input')
        )
        self.seg_win += 1
        self.vis.image(
            mask,
            win=self.seg_win,
            opts=dict(title='Sample Segmentation Label', caption='Label')
        )
        self.seg_win += 1
        self.vis.image(
            pred,
            win=self.seg_win,
            opts=dict(title='Sample Segmentation Pred', caption='Pred')
        )
        self.seg_win += 1
        
        
# # test functions
# vis = Visualizations('test')
# vis.plot_loss(np.array([0,1]),0)
# vis.plot_loss(np.array([2,3]),1)
# vis.plot_loss(np.array([4,5]),2)
# vis.plot_loss(np.array([1,3]),3)

# vis = Visualizations('test')
# img = sample['image']*0.5+0.5
# tmp_img = sample['image'].reshape(1,3,256,320)
# mask = sample['mask']
# pred = functional.softmax(model(tmp_img.cuda()), dim=1)
# pred_label = torch.max(pred,dim=1)[1]
# pred_label = pred_label.type(mask.type())
# vis.plot_image(img,class2mask(mask),class2mask(pred_label))