from pickletools import uint8
from turtle import Turtle
import cv2
import glymur

import numpy as np
import argparse
import torch
import os
import copy
from data.datasets import get_loader
from torchvision.utils import save_image
from datetime import datetime
import time
from utils import *
from loss.distortion import *
import torch.nn as nn
import warnings
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    CR_ini = 4 / 100  # CR of other methods

    SNR_dB = 4
    SNR = 10 ** (SNR_dB / 10)

    CR = CR_ini * (np.log2(1 + SNR)) / 8
    print('*' * 50)
    print('CR for JPEG2000:', CR)

    encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), 1000] 
    img = cv2.imread("raw.jpg")
    cv2.imwrite("compressed_image.jp2",img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(CR * 1000)])


    img = cv2.imread("compressed_image.jp2")
    image_tensor = torch.from_numpy(img).cuda()
    image_tensor = torch.transpose(image_tensor, 0, 2).clone()
    image_tensor = torch.transpose(image_tensor, 1, 2).clone()

    image_new = image_tensor.clone()
    image_new[0] = image_tensor[2].clone()
    image_new[2] = image_tensor[0].clone()

    image_tensor = (image_new - image_new.min()) / (image_new.max() - image_new.min())

    recon_image = torch.unsqueeze(image_tensor.clone(), dim=0)

    recon_image_down = downsampling(recon_image, 512)
    save_image(recon_image_down, 'JPEG2000_recovered.png') 

    