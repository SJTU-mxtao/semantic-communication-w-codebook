# train the codeword search NN, which has the same structure of classifier
import torch.optim as optim
from net.networkCB import WITT
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

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
    print_step = 100
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

    if args.trainset == 'CIFAR10':
        save_model_freq = 50  # save model epoch
        image_dims = (3, 32, 32)
        train_data_dir = "./media/Dataset/CIFAR10/"
        test_data_dir = "./media/Dataset/CIFAR10/"
        # batch_size = 128 
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    
    elif args.trainset == 'STL10':
        CBsize = 64 # 16, 32, 64
        batch_size = 32
        save_model_freq = 10  # save model epoch and results

        image_dims = (3, 256, 256)
        # image_dims = (3, 96, 96)
        train_data_dir = "./media/Dataset/CIFAR10/"
        test_data_dir = "./media/Dataset/CIFAR10/"
        # batch_size = 128 
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

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


def train_one_epoch(args):
    error_time = 0
    net.train()
    elapsed, losses, accs = [AverageMeter() for _ in range(3)]
    metrics = [elapsed, losses, accs]
    global global_step
    for batch_idx, (input, label) in enumerate(train_loader): 
        start_time = time.time()

        global_step += 1
        input = input.cuda()
        label = label.cuda()

        # search labels generated by the method of exhaustion
        codeword_ID = torch.ones(input.size()[0]).cuda()
        # search for the codeword
        ID_local = 1

        # MSE (alpha = 0)
        for image_ID in range(input.size()[0]):
            mse_ini = 10 ** 8
            for assist_ID in range(codebook.size()[0]):
                mse_local = MSE_loss(input[image_ID], codebook[assist_ID])
                if mse_local < mse_ini:
                    mse_ini = mse_local 
                    ID_local = assist_ID
            codeword_ID[image_ID] = codeword_ID[image_ID] * ID_local

        codeword_ID = codeword_ID.long()

        downsampled_image  = downsampling(input, 96)  # from 256 * 256 to 96 * 96
        out_class = classifier(downsampled_image)

        loss = CE_loss(out_class, codeword_ID)  # loss_G is the loss for classification

        _, pred = out_class.max(1)
        num_correct = (pred == codeword_ID).sum().item()
        acc = num_correct / input.shape[0] * 100

        optimizer_classifier.zero_grad()
        loss.backward() 
        optimizer_classifier.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        accs.update(acc)
        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Acc {accs.val:.3f} ({accs.avg:.3f})', 
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()
    
            val_Acc_all.append(accs.val)
            train_Acc_all.append(accs.avg)

    for i in metrics:
        i.clear()

def test(test_counter_local, datetime_proposed_local, datetime_exhaustion_local):
    config.isTrain = False
    net.eval()
    elapsed, accs = [AverageMeter() for _ in range(2)]
    metrics = [elapsed, accs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_acc = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(test_loader):
                start_time = time.time()
                input = input.cuda()
                label = label.cuda()

                # search labels generated by the method of exhaustion
                codeword_ID = torch.ones(input.size()[0]).cuda()
                # search for the codeword
                ID_local = 1
                test_counter_local += 1
         
                # search labels generated by the method of exhaustion
                codeword_ID = torch.ones(input.size()[0]).cuda()
                # search for the codeword
                ID_local = 1

                # WDS distance
                start_time_exhaustion = time.time()
                for image_ID in range(input.size()[0]):
                    mse_ini = 10 ** 8
                    for assist_ID in range(codebook.size()[0]):
                        recon_image_down  = downsampling(input, 96)  # from 256 * 256 to 96 * 96
                        out_class = classifier0(recon_image_down)

                        mse_local = MSE_loss(input[image_ID], codebook[assist_ID]) + 10 * CE_loss(out_class, label)
                        # mse_local = MSE_loss(input[image_ID], codebook[assist_ID])
                        if mse_local < mse_ini:
                            mse_ini = mse_local 
                            ID_local = assist_ID
                    codeword_ID[image_ID] = codeword_ID[image_ID] * ID_local
                end_time_exhaustion = time.time()
                datetime_exhaustion_local += (end_time_exhaustion - start_time_exhaustion)

                # print('Selected codeword:', codeword_ID)
                downsampled_image  = downsampling(input, 96)  # from 256 * 256 to 96 * 96

                start_time_proposed = time.time()
                out_class = classifier(downsampled_image)
                end_time_proposed = time.time()
                datetime_proposed_local += (end_time_proposed - start_time_proposed)

                _, pred = out_class.max(1)
                num_correct = (pred == codeword_ID).sum().item()
                acc = num_correct / input.shape[0] * 100

                accs.update(acc)

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'Acc {accs.val:.3f} ({accs.avg:.3f})', 
                ]))
                logger.info(log)

        results_acc[i] = accs.avg
        for t in metrics:
            t.clear()
        
        # add the testing results
        test_Acc_all.append(accs.avg)

    print("Acc: {}" .format(results_acc.tolist()))
    print("Finish Test!")

    return test_counter_local, datetime_exhaustion_local, datetime_proposed_local


def global_func():
    global train_Acc_all
    global val_Acc_all
    global test_Acc_all
    global datetime_proposed
    global datetime_exhaustion
    global test_counter

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    snr = int(args.multiple_snr.split(",")[0])

    net = WITT(args, config)

    datetime_proposed = 0
    datetime_exhaustion = 0
    test_counter = 0

    # load the codebook
    codebook_np = np.load('./results_data/SC_size' + str(config.CBsize) + '.npy')
    codebook = torch.from_numpy(codebook_np).cuda()
    codebook = codebook.view(config.CBsize, 3, 256, 256)

    # STL10/DIV2K should use the pre-trained model

    CE_loss = nn.CrossEntropyLoss()
    MSE_loss = nn.MSELoss()
    classifier = GoogLeNet(3, config.CBsize)  # number of category equals codebook size
    classifier.cuda()

    classifier0 = GoogLeNet(3, 10)  
    classifier0.load_state_dict(torch.load('google_net.pkl'))
    classifier0.cuda()

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=0.0001)

    train_Acc_all = []
    val_Acc_all = []
    test_Acc_all = []
    global_func()

    train_loader, test_loader = get_loader(args, config)

    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):          

            if epoch < 20:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
            elif epoch < 40:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
            elif epoch < 80:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
            elif epoch < 100:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
            elif epoch < 150:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
            elif epoch < 200:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
            else:
                optimizer = torch.optim.SGD(net.parameters(), lr=0.0002)

            train_one_epoch(args)

            # test()

            if (epoch + 1) % config.save_model_freq == 0:

                test_counter, datetime_exhaustion, datetime_proposed = test(test_counter_local=test_counter, datetime_exhaustion_local=datetime_exhaustion, datetime_proposed_local=datetime_proposed)  

                print('processing time of the exhaustion method:', datetime_exhaustion / test_counter)
                print('processing time of the proposed method:', datetime_proposed / test_counter)

                np_train_Acc_all = np.array(train_Acc_all)
                np_val_Acc_all = np.array(val_Acc_all)
                np_test_Acc_all = np.array(test_Acc_all)


                file = ('./results_data/results_SC/train_Acc_C%d_SNR%d_lambda%.1f.npy' % (args.C, snr, args.lambda_loss))
                np.save(file, np_train_Acc_all)

                file = ('./results_data/results_SC/val_Acc_C%d_SNR%d_lambda%.1f.npy' % (args.C, snr, args.lambda_loss))
                np.save(file, np_val_Acc_all)

                file = ('./results_data/results_SC/test_Acc_C%d_SNR%d_lambda%.1f.npy' % (args.C, snr, args.lambda_loss))
                np.save(file, np_test_Acc_all)

                torch.save(classifier.state_dict(), 'search_net_size' + str(config.CBsize) + '.pkl')


    else:
        test()

