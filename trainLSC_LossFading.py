import torch.optim as optim
from net.networkCBlossFading import WITT
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
from torchvision.utils import save_image
import time
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import math

parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='STL10',
                    choices=['CIFAR10', 'STL10'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='STL10',
                    choices=['CIFAR10', 'STL10'],
                    help='test dataset name')
parser.add_argument('--distortion-metric', type=str
                    , default='MSE',
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
parser.add_argument('--seed', type=int, default=1024,
                    help='random seed')
parser.add_argument('--SCsize', type=int, default=32,
                    choices=[10, 16, 32, 64],
                    help='SC size')
parser.add_argument('--lambda_loss', type=float, default='0.01',
                    help='lambda in the loss function')
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
        alpha = 100  # parameter in loss function, need to be adjust

        # CBsize = 32 # 10, 16, 32, 64
        save_model_freq = 10  # save model epoch and results

        image_dims = (3, 256, 256)
        # image_dims = (3, 96, 96)
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

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


def train_one_epoch(args, lambda_loss_local, H_fading_all):
    error_time = 0
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10' or args.trainset == 'STL10':
        for batch_idx, (input, label) in enumerate(train_loader):
            H_id = int(epoch * batch_idx) % 19999
            H_fading = H_fading_all[H_id]

            start_time = time.time()

            global_step += 1
            label_np = label.numpy()

            input = input.cuda()
            label = label.cuda()

            # search for the codeword
            code_assist = input.clone() 
            code_index = [] 
            for image_ID in range(input.size()[0]):
                l = label_np[image_ID]

                if args.SCsize == 32:
                    # SC size=32
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

                elif args.SCsize == 10:
                    # SC size=10
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


                code_index_local = 0
                mse_ini = 10 ** 8
                for assist_ID in LCB:
                    mse_local = MSE_loss(input[image_ID], codebook[assist_ID])
                    if mse_local < mse_ini:
                        code_assist[image_ID] = codebook[assist_ID].clone()
                        code_index_local = assist_ID
                        mse_ini = mse_local
                code_index.append(code_index_local)

            code_index = torch.from_numpy(np.array(code_index)).cuda()
            recon_image, CBR, SNR, mse, loss_G, loss_P = net(input, code_assist, code_index, H_fading)  # loss_G is the loss for generating image

            # loss_G = loss_G + config.alpha * loss_P
            loss = (loss_G - lambda_loss_local * loss_P).clone()

            if math.isnan(loss.item()): 
                print('Loss error! Please choose another lambda!')
                pdb.set_trace()
            
            if loss.item() >= 0:
                pass
            else:
                print('loss G:', loss_G.item())  
                print('loss P:', loss_P.item())
                # lambda_loss_local = lambda_loss_local / 2
                print('lambda_loss:', lambda_loss_local)
                loss = loss_G.clone()

            # update the coder NNs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if batch_idx % 50 == 0 and epoch % 5 == 0:
                # save image                  
                recon_image = downsampling(recon_image, 512)
                for iii in range(input.size()[0]):
                    if args.channel_type == 'awgn':
                        save_image(recon_image[iii], ('./image_recover_LSC_loss/img%d_epoch%d_batch%d_snr%d.png' % (iii, epoch, batch_idx, snr))) 
                    else:
                        save_image(recon_image[iii], ('./image_recover_LSC_loss_Fading/img%d_epoch%d_batch%d_snr%d.png' % (iii, epoch, batch_idx, snr)))  

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
                
                # add the training and validating results
                val_PSNR_all.append(psnrs.val)
                train_PSNR_all.append(psnrs.avg)
                val_SSIM_all.append(msssims.val)
                train_SSIM_all.append(msssims.avg)

    for i in metrics:
        i.clear()
    return lambda_loss_local

def test(H_fading_all):
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10' or args.trainset == 'STL10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    label_np = label.numpy()

                    
                    H_id = int(epoch * batch_idx) % 19999
                    H_fading = H_fading_all[H_id]  

                    input = input.cuda()
                    label = label.cuda()

                    # search for the codeword
                    code_assist = input.clone()  
                    code_index = []  
                    for image_ID in range(input.size()[0]):

                        l = label_np[image_ID]

                        if args.SCsize == 32:
                            # SC size=32
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

                        elif args.SCsize == 10:
                            # SC size=10
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
                        
                        code_index_local = 0
                        mse_ini = 10 ** 8
                        for assist_ID in LCB:
                            mse_local = MSE_loss(input[image_ID], codebook[assist_ID])
                            if mse_local < mse_ini:
                                code_assist[image_ID] = codebook[assist_ID].clone()
                                code_index_local = assist_ID
                                mse_ini = mse_local
                        # print('code_index_local', code_index_local)
                        code_index.append(code_index_local)

                    code_index = torch.from_numpy(np.array(code_index)).cuda()
                    recon_image, CBR, SNR, mse, loss_G, loss_P = net(input, code_assist, code_index, H_fading)  # loss_G is the loss for generating image

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)


                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    # logger.info(log)

        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()
        
        # add the testing results
        test_PSNR_all.append(psnrs.avg)
        test_SSIM_all.append(msssims.avg)

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")


def global_func():
    global train_PSNR_all
    global train_SSIM_all
    global val_PSNR_all
    global val_SSIM_all
    global test_PSNR_all
    global test_SSIM_all

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=args.seed)

    snr_ini = int(args.multiple_snr.split(",")[0])

    net = WITT(args, config)

    snr = int(args.multiple_snr.split(",")[0])

    # load the codebook
    codebook_np = np.load('./results_data/LSC_size' + str(args.SCsize) + '.npy')
    codebook = torch.from_numpy(codebook_np).cuda()
    codebook = codebook.view(args.SCsize, 3, 256, 256)

    if args.channel_type == 'awgn':
        pre_model_exist = False
    else:
        # use the pre-trained model for fading channel
        model_path = "./saved_model/awgn/STL10/LSC_loss_snr" + str(snr_ini) + "_C" + str(args.C) + ".model"
        pre_model_exist = os.path.isfile(model_path)  # if the pre-trained model exists
        if pre_model_exist:
            load_weights(model_path)
            print('*' * 50)
            print('load model parameters ...')

    CE_loss = nn.CrossEntropyLoss()
    MSE_loss = nn.MSELoss()
    classifier = GoogLeNet(3, 10)  
    classifier.load_state_dict(torch.load('google_net.pkl'))
    classifier.cuda()

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=0.0001)  # fine-tune the classifier, the learning rate should be very small

    train_PSNR_all = []
    train_SSIM_all = []
    val_PSNR_all = []
    val_SSIM_all = []
    test_PSNR_all = []
    test_SSIM_all = []
    global_func()

    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    # model_params = [{'params': net.parameters(), 'lr': 0.0005}]  # the higher learning rate, the smaller epoch
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)

    if args.channel_type == 'awgn':
        H_fading_all = np.zeros(20000)
    else:
        H_fading_all = np.sqrt(2 / np.pi) *  np.random.rayleigh(1, 20000)  # generate the fading coefficient
        H_fading_all = 10 * np.log10(H_fading_all + 10 ** (-10))

    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.training:
        lambda_loss = args.lambda_loss
        
        for epoch in range(steps_epoch, config.tot_epoch):      
            if epoch < 200:
                cur_lr = 0.01
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 400:
                cur_lr = 0.005
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 550:
                cur_lr = 0.002
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 650:
                cur_lr = 0.001
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 750:
                cur_lr = 0.0005
                optimizer = optim.Adam(model_params, lr=cur_lr)
            else:
                cur_lr = 0.0001
                optimizer = optim.Adam(model_params, lr=cur_lr)


            lambda_loss = train_one_epoch(args, lambda_loss)

            if (epoch + 1) % config.save_model_freq == 0:
                if args.channel_type == 'awgn':
                    save_model(net, save_path='./saved_model/awgn/STL10/LSC_loss_snr{}_C{}.model'.format(snr, args.C))
                else:
                    save_model(net, save_path='./saved_model/rayleigh/STL10/LSC_loss_snr{}_C{}.model'.format(snr, args.C))
                test(H_fading_all)

                np_train_PSNR_all = np.array(train_PSNR_all)
                np_train_SSIM_all = np.array(train_SSIM_all)
                np_val_PSNR_all = np.array(val_PSNR_all)
                np_val_SSIM_all = np.array(val_SSIM_all)
                np_test_PSNR_all = np.array(test_PSNR_all)
                np_test_SSIM_all = np.array(test_SSIM_all)


                if args.channel_type == 'awgn':
                    channel_str = 'results_LSC_loss'
                else:
                    channel_str = 'results_LSC_loss_Fading'

                file = ('./results_data/%s/train_PSNR_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_train_PSNR_all)

                file = ('./results_data/%s/train_SSIM_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_train_SSIM_all)

                file = ('./results_data/%s/val_PSNR_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_val_PSNR_all)

                file = ('./results_data/%s/val_SSIM_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_val_SSIM_all)

                file = ('./results_data/%s/test_PSNR_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_test_PSNR_all)

                file = ('./results_data/%s/test_SSIM_C%.2f_SNR%d_lambda%.2f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_test_SSIM_all)

    else:
        test(H_fading_all)

