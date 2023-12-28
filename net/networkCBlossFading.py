from math import nan
from net.decoderCB import *
from net.encoderCBloss import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn
import pdb
import torch
import numpy as np


class WITT(nn.Module):
    def __init__(self, args, config):
        super(WITT, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs) 
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, input_codebook, codeID, H_fading, given_SNR = None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W) 
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        SNR = None
        if given_SNR is None:
            SNR = choice(self.multiple_snr + H_fading)
            chan_param = SNR
        else:
            chan_param = given_SNR
        # encoding
        # feature = self.encoder(input_image, chan_param, self.model)  # torch.Tensor, size: [128, 64, 16] ([batch, 64, C]); see this as the mean of the Gaussian distribution
        feature, mu, sigma = self.encoder(input_image, input_codebook, chan_param, self.model)  # torch.Tensor, size: [128, 64, 16] ([batch, 64, C]); see this as the mean of the Gaussian distribution
        mu = mu.pow(2).clone()
        sigma_hat = sigma + torch.div(mu, 10 ** (SNR / 10))  # add the sigma of noise [batch, 1024]


        W = []  # selected codeword
        for imageID in range(mu.size()[0]):
            if codeID[imageID] in W:
                break
            else:
                W.append(codeID[imageID])

        loss_P = None 

        for i in range(len(W)):  # for each codeword W
            WX_counter = 0  # number of times the i-th W is selected in a batch
            mu_local = torch.zeros(1).cuda()  
            sigma_local = torch.zeros(1).cuda()
            for imageID in range(mu.size()[0]):
                if codeID[imageID] == W[i]:  # W is chosen by the imageID-th source image
                    WX_counter += 1 
                    mu_local = torch.add(mu_local, torch.sum(mu[imageID]) / 1024)
                    sigma_local = torch.add(sigma_local, torch.sum(sigma_hat[imageID]) / 1024)

            mu_local = mu_local / WX_counter 
            sigma_local = sigma_local / WX_counter  

            if i == 0:  
                loss_P = mu_local + sigma_local - torch.log(sigma_local + 10 ** (-10))  
            else:
                loss_P = loss_P + mu_local + sigma_local - torch.log(sigma_local + 10 ** (-10))
                
        loss_P = loss_P / len(W) 
        

        CBR = feature.numel() / 2 / input_image.numel()
        # Feature pass channel
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        # decoding
        recon_image = self.decoder(noisy_feature, input_codebook, chan_param, self.model)  # codebook index is transmitted in an error-free link
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))


        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean(), loss_P
