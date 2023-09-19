from __future__ import print_function

import argparse
import csv
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
from tqdm import tqdm
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import numpy as np
from model import HCNet
from Config import HyperParams
from pest_dataset import load_data
from plant_dataset import load_train_val_data


def parse_option():
    parser = argparse.ArgumentParser('Progressive Region Enhancement Network(PRENet) for training and testing')
    parser.add_argument('--batchsize', default=32, type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, default='food101')
    parser.add_argument('--image_path', type=str, default=r"D:\dataset\IP102\IP102_txt/", help='path to dataset')
    parser.add_argument("--train_path", type=str, default=r"D:\dataset\IP102/train.txt", help='path to training list')
    parser.add_argument("--test_path", type=str, default=r"D:\dataset\IP102/test.txt",
                        help='path to testing list')
    parser.add_argument('--weight_path', default=r"C:\Users\ljw\PycharmProjects\pythonProject\fine-grained image recognition\Large Scale Visual Food Recognition\prenet-master\prenet\food2k_resnet50_0.0001.pth", help='path to the pretrained model')
    parser.add_argument('--use_checkpoint', action='store_true', default=True,
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--checkpoint', type=str, default=r"C:\Users\ljw\PycharmProjects\pythonProject\fine-grained image recognition\Large Scale Visual Food Recognition\prenet-master\model_448_from2k\model.pth",
                        help="the path to checkpoint")
    parser.add_argument('--output_dir', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--epoch", default=150, type=int,
                        help="The number of epochs.")
    args, unparsed = parser.parse_known_args()
    return args
def train(epochs, train_loader, valloader, model_name):

    net = HCNet(class_num=102, arch=model_name)
    model_weight_path = "./"
    net.load_state_dict(torch.load(model_weight_path))
    net = net.cuda()
    netp = nn.DataParallel(net).cuda()
    CELoss = nn.CrossEntropyLoss()
    ########################
    new_params, old_params = net.get_params()
    # --------------------------------------------------------------------------------------------------------------------------------#SGD
    new_layers_optimizer = optim.SGD(new_params, momentum=0.9, weight_decay=5e-4, lr=0.002)
    old_layers_optimizer = optim.SGD(old_params, momentum=0.9, weight_decay=5e-4, lr=0.0002)
    new_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_layers_optimizer, HyperParams['epoch'], 0)
    old_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_layers_optimizer, HyperParams['epoch'], 0)
    max_val_acc = 0
    f = open('./ablation_result/HCNet_convnext__test.csv', 'w', newline='')
    write_csv = csv.writer(f)
    write_csv.writerow(['HCNet_convnext_base_convnex_CAFC1_test'])

    for epoch in range(0, epochs):
        print('\nEpoch: %d' % epoch)
        start_time = datetime.now()
        print("start time: ", start_time.strftime('%Y-%m-%d-%H:%M:%S'))
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0

        train_loader = tqdm(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            idx = batch_idx
            inputs, targets = inputs, targets.cuda()
      # ----------------------------------------#
            output_1, output_2, output_3 = netp(inputs)

            # adjust optimizer lr
            new_layers_optimizer_scheduler.step()
            old_layers_optimizer_scheduler.step()

            # overall update
            loss1 = CELoss(output_1, targets)*2
            loss2 = CELoss(output_2, targets)*2
            loss3 = CELoss(output_3, targets)*2


            new_layers_optimizer.zero_grad()
            old_layers_optimizer.zero_grad()

            loss = loss1 + loss2 + loss3
            loss.backward()
            new_layers_optimizer.step()
            old_layers_optimizer.step()

            #  training log
            _, predicted = torch.max((output_1+output_2+output_3).data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            train_loss += (loss1.item() + loss2.item() + loss3.item() )
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()


            if batch_idx % 50 == 0:
                print('Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        # eval
        val_acc = test(net, valloader)
        write_csv.writerow([round(val_acc,4)])

        if val_acc  > max_val_acc:
            max_val_acc = val_acc
            torch.save(net.state_dict(), './plant_file/'+model_name+'model.pth')
        print("best result: ", max_val_acc)


        print("current result: ", val_acc)
        torch.save(net.state_dict(), './plant_file/' + model_name + 'model.pth')
        end_time = datetime.now()
        print("end time: ", end_time.strftime('%Y-%m-%d-%H:%M:%S'))

def test(net, testloader):
    net.eval()
    correct_com = 0
    total = 0

    softmax = nn.Softmax(dim=-1)
    testloader = tqdm(testloader)
    for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                output_1, output_2, output_3 = net(inputs)
                outputs_com = output_1 + output_2 + output_3

            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

    return test_acc_com

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    args = parse_option()
    set_seed(2023)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # train_dataset, train_loader, test_dataset, test_loader = load_data(image_path=args.image_path, train_txt=args.train_path, val_txt=args.test_path,batch_size=args.batchsize)
    train_dataset, train_loader, test_dataset, test_loader = load_train_val_data() #plant
    train(epochs=150, train_loader=train_loader, valloader=test_loader,model_name='convnext_base')
