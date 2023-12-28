from pickletools import uint8
from turtle import Turtle
import cv2

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='STL10',
                    choices=['CIFAR10', 'STL10'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='STL10',
                    choices=['CIFAR10', 'STL10'],
                    help='test dataset name')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=32,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
parser.add_argument('--lambda_loss', type=float, default='0.01',
                    help='lambda in the loss function, 0.01, 0.1, 1, 10, 100')
args = parser.parse_args()



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



class config():
    seed = 1024  # random seed
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:3")
    norm = False
    # logger
    print_step = 1000
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    # learning_rate = 0.0005

    # tot_epoch = 10000000
    tot_epoch = 500000

    CBsize = 64 # 8, 16, 32, 64
    save_model_freq = 10  # save model epoch and results

    image_dims = (3, 256, 256)
    train_data_dir = "./media/Dataset/CIFAR10/"
    test_data_dir = "./media/Dataset/CIFAR10/"
    # batch_size = 128 
    batch_size = 12
    downsample = 4
    encoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    decoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


def test():
    config.isTrain = False
    elapsed, losses, psnrs, msssims, cbrs, snrs, accs = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs, accs]
    global global_step

    for batch_idx, (input, label) in enumerate(train_loader):

        start_time = time.time()
        global_step += 1

        input = input.cuda()
        label = label.cuda()

        recon_image = copy.deepcopy(input)
        for image_ID in range(np.shape(input)[0]):
            save_image(input[image_ID], "img.jpg")

            H_id = int(epoch * batch_idx * image_ID) % 19999
            CR_local = CR_new[H_id]
            snr = snr_new[H_id]

            snrs.update(snr)

            # JPEG2000 coding based on cv2
            # you can choose the traditional image compression method by replacing the codes enclosed in asterisks 
            # When visualization of results is needed, simply use the codes enclosed in asterisks to process the corresponding image in `image_raw`.
            # *******************************************************************************************************
            encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), 1000] 
            img = cv2.imread("img.jpg")
            # cv2.imwrite("compressed_image.jp2", img, encode_param)
            cv2.imwrite("compressed_image.jp2",img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(CR_local * 1000)])
            # *******************************************************************************************************

            img = cv2.imread("compressed_image.jp2")
            image_tensor = torch.from_numpy(img).cuda()
            image_tensor = torch.transpose(image_tensor, 0, 2).clone()
            image_tensor = torch.transpose(image_tensor, 1, 2).clone()

            image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

            image_new = image_tensor.clone()
            image_new[0] = image_tensor[2].clone()
            image_new[2] = image_tensor[0].clone()

            recon_image[image_ID] = image_new.clone()

            # recon_image_down = downsampling(recon_image, 512)
            # for iii in range(input.size()[0]):
            #     save_image(recon_image_down[iii], ('./image_recover_JPEG2000/img%d_epoch%d_batch%d_snr%d.png' % (iii, epoch, batch_idx, snr))) 

        input = input 
        recon_image = recon_image 
        mse = MSE_loss(input, recon_image)

        recon_image_down  = downsampling(recon_image, 96)  # from 256 * 256 to 96 * 96
        out_class = classifier(recon_image_down)

        _, pred = out_class.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / input.shape[0] * 100

        # loss_C = CE_loss(out_class, label)
        # loss_C = loss_C.requires_grad_()

        # optimizer_classifier.zero_grad()
        # loss_C.backward() 
        # optimizer_classifier.step()

        elapsed.update(time.time() - start_time)
        if mse.item() > 0:
            psnr = 10 * (torch.log(1. / mse) / np.log(10))
            psnrs.update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)
            accs.update(acc)
        else:
            psnrs.update(100)
            msssims.update(100)
            accs.update(100)

        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'SNR {snrs.val:.3f}',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Acc {accs.val:.3f} ({accs.avg:.3f})', 
            ]))
            logger.info(log)

            for t in metrics:
                t.clear()

            # add the testing results
            test_Acc_all.append(accs.avg)
            test_PSNR_all.append(psnrs.avg)
            test_SSIM_all.append(msssims.avg)



def global_func():
    global test_PSNR_all
    global test_Acc_all
    global test_SSIM_all
    global snr_new
    global CR_new


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)

    # JPEG2000 v.s. different CRs
    CR_ini = np.array([1, 2, 3, 4, 5]) / 100  
    SNR_dB = np.array([10, 10, 10, 10, 10])

    # JPEG2000 v.s. different SNRs
    # CR_ini = np.array([4, 4, 4, 4]) / 100 
    # SNR_dB = np.array([2, 4, 6, 8])

    SNR = 10 ** (SNR_dB / 10)

    CR = CR_ini / 8

    pre_model_exist = False

    CE_loss = nn.CrossEntropyLoss()
    MSE_loss = nn.MSELoss()
    classifier = GoogLeNet(3, 10)  
    classifier.load_state_dict(torch.load('google_net.pkl'))
    classifier.cuda()

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=0.0001)  # Classifier is also fine-tuned in JPEG2000 method

    test_PSNR_all = []
    test_Acc_all = []
    test_SSIM_all = []

    train_loader, test_loader = get_loader(args, config)
    del test_loader

    global_step = 0

    if args.channel_type == 'awgn':
        H_fading_all = np.ones(20000)
    else:
        H_fading_all = np.sqrt(2 / np.pi) *  np.random.rayleigh(1, 20000)  # generate the fading coefficient

    for cr in range(np.shape(CR)[0]):
        CR_local = CR[cr]
        snr_ini = SNR[cr]

        snr = SNR[cr]
        print('*' * 100)
        print('Compression Rate:', CR_local)

        snr_new = copy.deepcopy(SNR_dB[cr] + 10 * np.log10(H_fading_all + 10 ** (-10)))
        CR_new  = copy.deepcopy(CR_local * (np.log2(1 + (snr * H_fading_all))))

        global_func()
        
        global_step = 0
        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(20): 

            test()

            np_test_PSNR_all = np.array(test_PSNR_all)
            np_test_Acc_all = np.array(test_Acc_all)
            np_test_SSIM_all = np.array(test_SSIM_all)

            file = ('./results_data/results_JPEG2000/test_PSNR_C%d_SNR%d_lambda%.1f.npy' % (CR_local, snr_ini, args.lambda_loss))
            np.save(file, np_test_PSNR_all)

            file = ('./results_data/results_JPEG2000/test_Acc_C%d_SNR%d_lambda%.1f.npy' % (CR_local, snr_ini, args.lambda_loss))
            np.save(file, np_test_Acc_all)

            file = ('./results_data/results_JPEG2000/test_SSIM_C%.2f_SNR%d_lambda%.2f.npy' % (CR_local, snr_ini, args.lambda_loss))
            np.save(file, np_test_SSIM_all)









