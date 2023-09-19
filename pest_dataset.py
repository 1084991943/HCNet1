import torch
import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

def Convert_RGB(path):
    return PIL.Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, image_path,transform = None):
        data_txt = open(txt_path, 'r')
        image = []
        for line in data_txt:
            line = line.strip()
            name_ , class_ = line.split('\t')
            image.append(( name_,  int(class_)))
        self.imgs = image
        self.Convert_RGB = Convert_RGB
        self.img_path = image_path
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img_name_, target = self.imgs[index]
        img = self.Convert_RGB(self.img_path+ img_name_+ '.jpg')
        # img = self.img_path+ img_name_+ '.jpg'
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def load_data(image_path, train_txt, val_txt, batch_size=32, num_workers=8, input_size= (224,224)):
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
    train_dataset = MyDataset(txt_path=train_txt,image_path=image_path,transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_dataset = MyDataset(txt_path=val_txt, image_path=image_path, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, train_loader, val_dataset, val_loader

def load_data_test(image_path,test_txt, batch_size=32, num_workers=8, input_size= (224,224)):

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                             std=[0.23273608, 0.24383051, 0.24237761])])

    test_dataset = MyDataset(txt_path=test_txt, image_path=image_path, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)
    return test_dataset, test_loader
