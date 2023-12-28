from unicodedata import category
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import warnings
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


warnings.filterwarnings("ignore")


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

        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

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
        x = self.block1(x)
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

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def data_tf(x):
    x = x.resize((256, 256), 2) 
    x = np.array(x, dtype='float32') / 255
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    train_set = datasets.STL10(root=config.stl_path, transform=data_tf, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = datasets.STL10(root=config.stl_path, transform=data_tf, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader


def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


if __name__ == '__main__':

    CBsize = 10  # codebook size, 10 16 32 64  
    lambda_loss = 10  # lambda in loss, 0.1 1 10 100
    

    torch.manual_seed(1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)   
    batchsize = CBsize
    epoch_len = 500
    codebook = None
    distance_measure1 = nn.MSELoss()
    distance_measure2 = nn.CrossEntropyLoss()

    counter = np.ones(CBsize)
    iteration_interval = 10

    train_set = datasets.STL10("./media/Dataset/CIFAR10/", transform=data_tf, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True, drop_last=False)

    classifier = GoogLeNet(3, 10)  
    classifier.load_state_dict(torch.load('google_net.pkl'))
    classifier.to(device)

    print('Codebook Construction Start!')
    
    for e in range(epoch_len):
        print('epoch:', e)
        iteration = 0
        for im, label in train_loader:
            iteration += 1

            label_np = label.numpy()

            im = Variable(im) 
            label = Variable(label)  
            im = im.to(device)
            label = label.to(device)  # batchsize

            # initialize the codebook
            if e == 0:
                codebook = im.clone()
                print('codebook initialization is done ...')
                break

            category = 0  # the corresponding codeword index

            codebook_down = downsampling(codebook, 96)
            out_codebook = classifier(codebook_down)  # batchsize, class_num
            
            for i in range(batchsize):
                distance_min = 10 ** 8

                l = label_np[i]  # class label
                if CBsize == 16:
                    if l == 0:      
                        LCB = np.array([0, 1])
                    elif l == 1:
                        LCB = np.array([2, 3])
                    elif l == 2:
                        LCB = np.array([4, 5])
                    elif l == 3:
                        LCB = np.array([6, 7])
                    elif l == 4:
                        LCB = np.array([8, 9])
                    elif l == 5:
                        LCB = np.array([10, 11])
                    elif l == 6:
                        LCB = np.array([12])
                    elif l == 7:
                        LCB = np.array([13])
                    elif l == 8:
                        LCB = np.array([14])
                    else:
                        LCB = np.array([15])

                elif CBsize == 32:
                    if l == 0:      
                        LCB = np.array([0, 1, 2, 3])
                    elif l == 1:
                        LCB = np.array([4, 5, 6, 7])
                    elif l == 2:
                        LCB = np.array([8, 9, 10])
                    elif l == 3:
                        LCB = np.array([11, 12, 13])
                    elif l == 4:
                        LCB = np.array([14, 15, 16])
                    elif l == 5:
                        LCB = np.array([17, 18, 19])
                    elif l == 6:
                        LCB = np.array([20, 21, 22])
                    elif l == 7:
                        LCB = np.array([23, 24, 25])
                    elif l == 8:
                        LCB = np.array([26, 27, 28])
                    else:
                        LCB = np.array([29, 30, 31])

                elif CBsize == 64:
                    if l == 0:      
                        LCB = np.array([0, 1, 2, 3, 4, 5, 6])
                    elif l == 1:
                        LCB = np.array([7, 8, 9, 10, 11, 12, 13])
                    elif l == 2:
                        LCB = np.array([14, 15, 16, 17, 18, 19, 20])
                    elif l == 3:
                        LCB = np.array([21, 22, 23, 24, 25, 26, 27])
                    elif l == 4:
                        LCB = np.array([28, 29, 30, 31, 32, 33])
                    elif l == 5:
                        LCB = np.array([34, 35, 36, 37, 38, 39])
                    elif l == 6:
                        LCB = np.array([40, 41, 42, 43, 44, 45])
                    elif l == 7:
                        LCB = np.array([46, 47, 48, 49, 50, 51])
                    elif l == 8:
                        LCB = np.array([52, 53, 54, 55, 56, 57])
                    else:
                        LCB = np.array([58, 59, 60, 61, 62, 63])
                
                elif CBsize == 10:
                    if l == 0:      
                        LCB = np.array([0])
                    elif l == 1:
                        LCB = np.array([1])
                    elif l == 2:
                        LCB = np.array([2])
                    elif l == 3:
                        LCB = np.array([3])
                    elif l == 4:
                        LCB = np.array([4])
                    elif l == 5:
                        LCB = np.array([5])
                    elif l == 6:
                        LCB = np.array([6])
                    elif l == 7:
                        LCB = np.array([7])
                    elif l == 8:
                        LCB = np.array([8])
                    else:
                        LCB = np.array([9])
                
                else:
                    print('wrong SC size!')
                    pdb.set_trace()

                
                for j in LCB:  # select the codeword with the same class label
                    distance1 = distance_measure1(im[i], codebook[j])
                    
                    distance2 = distance_measure2(out_codebook[j].unsqueeze(0).repeat(2, 1), label[i].repeat(2))  # batch size for a loss function should not equal one
                    distance = distance1 + lambda_loss * distance2
                    
                    if distance < distance_min:
                        distance_min = distance
                        category = j
                counter[category] += 1
                codebook[category] = torch.add(codebook[category] * (counter[category] - 1) / counter[category], im[i] / counter[category])

            if iteration == iteration_interval: 
                break
        
        print('counters:', counter)

        if e % 10 == 0:
            print('save the codebook ...')
            codebook0 = codebook.clone()
            codebook0 = codebook0.view(CBsize, int(3 * 256 *256))
            np_codebook = codebook0.detach().cpu().numpy()

            file = ('./results_data/LSC_size%d.npy' % (CBsize))
            np.save(file, np_codebook)




                