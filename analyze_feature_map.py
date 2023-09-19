import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
from model import HCNet

img = './img/02074.jpg'
img = Image.open(img).convert('RGB')

device = torch.device("cuda:0")
data_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5457954, 0.44430383, 0.34424934],
                                                          [0.23273608, 0.24383051, 0.24237761])])

net = HCNet(class_num=102, arch='convnext_base')
weight = './'
net.load_state_dict(torch.load(weight))
net.to(device)
print(net)
input_img = data_transform(img).unsqueeze(0)
print(input_img.shape)

activation = {}
def get_activation(name):
    def hook(net, input, output):
        activation[name] = output.detach()
    return hook

net.eval()
net.conv_block3[1].register_forward_hook(get_activation('relu'))
_ = net(input_img.to(device))
relu = activation['relu'].cpu().numpy()
for i in range(200):
    plt.imshow(relu[0,i,:,:], cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(r'C:\Users\ljw\Desktop\CEA_pest_2023_6_24\feature_map\with_CAFC\F3/' + str(i) + '.jpg',bbox_inches='tight', pad_inches = -0.1)
    plt.axis('off')

