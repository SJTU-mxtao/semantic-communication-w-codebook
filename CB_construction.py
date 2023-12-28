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


warnings.filterwarnings("ignore")


def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


def data_tf(x):
    x = x.resize((96, 96), 2) 
    x = np.array(x, dtype='float32') / 255
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


if __name__ == '__main__':
    CBsize = 64  # codebook size, 8 16 32 64

    torch.manual_seed(1024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)   
    batchsize = CBsize
    epoch_len = 500
    codebook = None
    distance_measure = nn.MSELoss()
    counter = np.ones(CBsize)
    iteration_interval = 50

    train_set = datasets.STL10("./media/Dataset/CIFAR10/", transform=data_tf, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)

    print('Codebook Construction Start!')
    
    
    for e in range(epoch_len):
        print('epoch:', e)
        iteration = 0
        for im, label in train_loader:
            iteration += 1
            im = Variable(im)  # batchsize
            im = downsampling(im, 256)
            im = im.to(device)
            # initialize the codebook
            if e == 0:
                codebook = im.clone()
                print('codebook initialization is done ...')
                break

            distance_min = 10 ** 8
            category = 0  # the corresponding codeword index

            for i in range(batchsize):
                for j in range(CBsize):
                    distance = distance_measure(im[i], codebook[j])
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
            codebook0 = codebook0.view(CBsize, int(3 * 256 * 256))
            np_codebook = codebook0.detach().cpu().numpy()

            file = ('./results_data/codebook_size%d.npy' % (CBsize))
            np.save(file, np_codebook)




                