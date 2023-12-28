
# -*- coding: utf-8 -*-
import sys

sys.path.append("...")

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
# from torchvision.datasets import CIFAR10
from torchvision import datasets
import torch.nn.functional as F
from datetime import datetime
import os
import argparse
import pdb



def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # 第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # 第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # 第三条线路
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # 第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(GoogLeNet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)  # batch, 64, 23, 23
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)  # batch,1024
        x = self.classifier(x)
        return x

# test_net = GoogLeNet()
# test_x = Variable(torch.zeros(1, 3, 96, 96))
# test_y = test_net(test_x)
# print('output: {}'.format(test_y.shape))


def data_tf(x):
    # x <class 'PIL.Image.Image'>
    # x (32, 32, 3)
    x = x.resize((96, 96), 2)  # 将图片放大到96*96 shape of x: (96, 96, 3)
    x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))  ## 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x


def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    # transform = transforms.Compose([
    #     transforms.Scale(config.image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # cifar = datasets.CIFAR10(root=config.cifar_path, download=True, transform=transform)
    # stl = datasets.STL10(root=config.stl_path, download=True, transform=transform)

    train_set = datasets.STL10(root=config.stl_path, transform=data_tf, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = datasets.STL10(root=config.stl_path, transform=data_tf, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # stl = datasets.STL10(root=config.stl_path, download=True, transform=data_tf)

    # stl_loader = torch.utils.data.DataLoader(dataset=stl,
    #                                            batch_size=config.batch_size,
    #                                            shuffle=True,
    #                                            num_workers=config.num_workers)

    

    return train_loader, test_loader




def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()

        # adaptive learning rate
        # 学习率好像有点儿太大了。。。
        if epoch < 20:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
        elif epoch < 40:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        elif epoch < 80:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
        elif epoch < 120:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
        elif epoch < 150:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)

        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # batch, 3, 96, 96
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)

                # # backward
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str)
        torch.save(net.state_dict(), 'google_net.pkl')


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=96)
parser.add_argument('--num_classes', type=int, default=10)

# misc
parser.add_argument('--stl_path', type=str, default="./media/Dataset/CIFAR10/")

config = parser.parse_args()
print(config)

# model and
train_data, test_data = get_loader(config)
net = GoogLeNet(3, 10)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

train(net, train_data, test_data, 200, criterion)


