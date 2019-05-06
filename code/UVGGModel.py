import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
class UVGGModel(nn.Module):
	def __init__(self, num_class):
		super().__init__()
		pretrained_net = models.vgg16(pretrained = True).features
		self.conv1 = nn.Sequential(pretrained_net[0:5])
		self.conv2 = nn.Sequential(pretrained_net[5:10])
		self.conv3 = nn.Sequential(pretrained_net[10:17])
		self.conv4 = nn.Sequential(pretrained_net[17:24])
		self.conv5 = nn.Sequential(pretrained_net[24:])
		self.conv5m = add_conv_stage(512, 512, useBN=False)
		self.conv4m = add_conv_stage(1024, 512, useBN=False)
		self.conv3m = add_conv_stage(512, 256, useBN=False)
		self.conv2m = add_conv_stage(256, 128, useBN=False)
		self.conv1m = add_conv_stage(128,  64, useBN=False)
		self.upsample54 = upsample(512, 512)
		self.upsample43 = upsample(512, 256)
		self.upsample32 = upsample(256, 128)
		self.upsample21 = upsample(128, 64)
		self.upsample10 = upsample(64, 	64)
		self.classifier = add_conv_stage(64, num_class, kernel_size=1, padding=0, useBN=False)

	def forward(self, x):
		conv1_out = self.conv1(x)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		conv4_out = self.conv4(conv3_out)
		conv5_out = self.conv5(conv4_out)


		conv5m_out_ = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
		conv4m_out = self.conv4m(conv5m_out_)

		conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
		conv3m_out = self.conv3m(conv4m_out_)

		conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
		conv2m_out = self.conv2m(conv3m_out_)

		conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
		conv1m_out = self.conv1m(conv2m_out_)
		output_ = self.upsample10(conv1m_out)
		output = self.classifier(output_)
		return output


def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )
if __name__=="__main__":
	model = UVGGModel(num_class=11)
	x = torch.ones([1,3,320,256])
	print(model(x).shape)