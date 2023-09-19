import json
import os.path

import torch
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import json
def Convert_RGB(path):
    return Image.open(path).convert('RGB')

class Plantdoc_dataset(data.Dataset):
    def __init__(self, image_path, image_txt, class_json, transform=None):
        dataset_txt = open(image_txt, 'r')
        with open(class_json, 'r') as f:
            class_idx = json.load(f)
        image = []
        for line in dataset_txt:
            line = line.strip()
            name_, class_ = line.split('\t')
            class_index = int(class_idx[class_])
            image.append((name_, class_index, class_))
        self.images = image
        self.Convert_RGB = Convert_RGB
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        image_name, disease_target, file_name = self.images[item]
        # print(image_name,disease_target)
        # img = self.Convert_RGB(self.image_path + '\\'+ file_name +'\\' +image_name)
        img = self.Convert_RGB(os.path.join(self.image_path, file_name,image_name))
        img = self.transform(img)

        return img, disease_target



def load_train_val_data(dataset_name='AI2018',image_path = R'D:\dataset\AiChallenger',
                        train_txt= r'./plant_file/AI2018_train.txt', val_txt=r'./plant_file/AI2018_test.txt',
                        class_json =r'./plant_file/AI2018_class.json', batch_size=32, num_workers=8, input_size=(224,224)):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.Resize((256,256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                             std=[0.23273608, 0.24383051, 0.24237761])
    ])


    val_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                             std=[0.23273608, 0.24383051, 0.24237761])
    ])
    if dataset_name == 'AI2018':
        train_path = os.path.join(image_path, "train")
        val_path = os.path.join(image_path, "test")
        train_dataset = Plantdoc_dataset(image_path=train_path, image_txt= train_txt, class_json=class_json, transform=train_transform)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle= True, num_workers=num_workers)
        val_dataset = Plantdoc_dataset(image_path=val_path, image_txt=val_txt, class_json=class_json, transform=val_transform)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataset, train_loader, val_dataset, val_loader


