import os

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils_cam import GradCAM, show_cam_on_image, center_crop_img
from model import HCNet

def main():

    model = HCNet(class_num=102, arch='convnext_base')
    model_weight_path = r''
    model.load_state_dict(torch.load(model_weight_path),strict=False)
    model.cuda()
    target_layers = [model.block3]


    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224,224)),
                                         transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],std=[0.23273608, 0.24383051, 0.24237761])])

    save_dir = r'C:\Users\ljw\Desktop\CEA\cam_map\ConvNeXt\F3'
    # load image
    dir_path = R'D:\dataset\IP102\IP102\train'
    dir_list = os.listdir(dir_path)


    print(dir_list)
    for i in dir_list:
        img_dir = os.path.join(dir_path,i)
        image_name = os.listdir(img_dir)
        for j in image_name:
            img_path = os.path.join(img_dir, j)

            # img_path = R"D:\dataset\IP102\IP102\train\69/51296.jpg"

            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            # img = center_crop_img(img, 224)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

            target_category = None# tabby, tabby cat
            # target_category = 254  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor.cuda(), target_category=target_category)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                              grayscale_cam,
                                              use_rgb=True)
            visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_dir+'\\'+j, visualization)
            # plt.imshow(visualization)
            # plt.show()


if __name__ == '__main__':
    main()
