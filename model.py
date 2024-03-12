from utils import *
import torch.nn as nn
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )


    def forward(self, x):

        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape, self.module(x).shape)
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self,num_psfs):
        super(RCAN, self).__init__()
        scale = 2
        num_features = 32
        num_rg = 2
        num_rcab = 3
        reduction = 16
        self.downscale = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(num_psfs*scale ** 2, num_features, kernel_size=3, padding=1)
        )
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.downscale(x)
        residual = x
        x = self.rgs(x)
        # print(x.shape)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class FourierDeconvolution2D_ds(nn.Module):
    """
    Performs Deconvolution in the frequency domain for each psf.

    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """

    def __init__(self,num_psfs, ps):
        super(FourierDeconvolution2D_ds, self).__init__()
        self.scale = ps
        self.channel = num_psfs
        self.psfs_re = nn.Parameter(torch.rand(self.channel, 4200//self.scale, (2100//self.scale)+1) * 0.001)
        self.psfs_im = nn.Parameter(torch.rand(self.channel, 4200//self.scale, (2100//self.scale)+1) * 0.001)
        self.ds =  nn.PixelUnshuffle(self.scale)
        self.us = nn.PixelShuffle(self.scale)
        self.conv = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1)
        torch.nn.init.normal_(self.conv.weight)
        self.activation = nn.PReLU()

    def forward(self, y):
        # preprocessing, Y is shape(batchsize, H, W)
        # psfs is shape(C, H, W)
        y = y.unsqueeze(1)
        y = self.ds(y)
        Y = torch.fft.rfft2(y, dim=(-2, -1))
        Y = Y.unsqueeze(1)
        psfs_re = self.psfs_re[None, ...]
        psfs_im = self.psfs_im[None, ...]
        psf_freq = torch.complex(psfs_re, psfs_im)
        X = Y * psf_freq.unsqueeze(2)
        x = torch.fft.irfft2(X, dim=(-2, -1))
        x = self.us(x).squeeze(2)
        return x

    def get_config(self):
        config = {
            'scale': self.scale,
            'channel': self.channel,
        }
        return config

class MultiWienerDeconvolution2D(nn.Module):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.

    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """

    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution2D, self).__init__()
        initial_psfs = torch.tensor(initial_psfs, dtype=torch.float32)
        initial_Ks = torch.tensor(initial_Ks, dtype=torch.float32)
        self.psfs = nn.Parameter(initial_psfs, requires_grad=True)
        self.Ks = nn.Parameter(initial_Ks, requires_grad=True)  # NEEED RELU CONSTRAINT HERE K is constrained to be nonnegative

    def forward(self, y):
        # Y preprocessing, Y is shape (B, C,H, W)
        y = y.unsqueeze(1)
        y = y.type(torch.complex64)
        # Temporarily transpose y since we cannot specify axes for fft2d
        Y = torch.fft.fft2(y)
        # Components preprocessing, psfs is shape (C,H, W)
        psf = self.psfs.type(torch.complex64)
        H_sum = torch.fft.fft2(psf)
        X = (torch.conj(H_sum) * Y) / (torch.square(torch.abs(H_sum)) + self.Ks)  # , dtype=tf.complex64)
        x = torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config


class LSVEnsemble2d(nn.Module):
    def __init__(self, deconvolution,enhancement):
        super(LSVEnsemble2d, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.deconvolution = deconvolution
        self.enhancement = enhancement

    def forward(self, x):
        initial_output = self.deconvolution(x)
        w = initial_output.shape[-1]
        h = initial_output.shape[-2]
        initial_output = initial_output / torch.max(initial_output)
        initial_output = initial_output[..., h//2+1 - 2400 // 2:h//2+1 + 2400 // 2, w//2+1 - 2400 // 2:w//2+1 + 2400 // 2]
        initial_output = initial_output / torch.max(initial_output)
        final_output = self.enhancement(initial_output)
        return final_output


class resblock(nn.Module):
    def __init__(self, channels=48):
        super(resblock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.act(self.bn1(x1))
        x1 = self.conv2(x1)
        return (self.bn2(x1))


class cm2netblock(nn.Module):
    def __init__(self, inchannels, numblocks, outchannels = 48):
        super(cm2netblock, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.numblocks = numblocks

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.resblocks = nn.ModuleList([resblock(self.outchannels) for i in range(numblocks)]) #If resblock class is defined in the same file, make sure it is defined before the cm2netblock class. If it is defined in a different file, ensure that import it using the appropriate import statement.
        self.conv2 = nn.Conv2d(outchannels, inchannels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self,x):  # check this, is the /1.414 still necessary since we're not doing the refocusing and enhancement branch anymore for now since only working with 2D
        x0 = self.act(self.conv1(x))
        x1 = torch.clone(x0)
        for _, modulee in enumerate(self.resblocks):
            x1 = (modulee(x1) + x1) / 1.414  # adding back after each res block and normalizing (arrow and plus sign in the diagram)
        x1 = (x1 + x0) / 1.414  # from input of 1st res block to output of last
        return self.conv2(x1)


class cm2net(nn.Module):
    def __init__(self, numBlocks, stackchannels=9, outchannels=48): # set the default argument for outchannels to be 1 but it's really set in each net()
        super(cm2net, self).__init__()
        self.demix = cm2netblock(stackchannels, numblocks=numBlocks, outchannels = outchannels)
        self.recon = cm2netblock(stackchannels, numblocks=numBlocks, outchannels = outchannels)
        self.endconv = nn.Conv2d(stackchannels, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, stack):
        # print(stack.shape)
        demix_result = self.activation(self.demix(stack))
        # print(demix_result.shape)
        output = self.recon(demix_result)  # no squeeze
        # print(output.shape)
        output = self.activation(self.endconv(output))
        return demix_result,output
