import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ExpCosGenerator(nn.Module):
    def __init__(self, n_channels, preprocess=None, gpu=False, N_Z=9408, lmbd=0.05):
        super(ExpCosGenerator, self).__init__()
        self.gpu = gpu
        self.preprocess = preprocess
        self.N_Z = N_Z
        self.lmbd=lmbd

        if self.N_Z == 128:
            self.ENC_SHAPE = (8, 4, 4)
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=8, out_channels=48, kernel_size=4, stride=4, padding=1),
                nn.LeakyReLU()
            )
        elif self.N_Z == 9408:
            self.ENC_SHAPE = (48, 14, 14)
        else:
            print("Hidden size %d is not implemented" %(self.N_Z))
            assert 0

        self.up1 = up(48, 48)
        self.up2 = up(48, 48)
        self.up3 = up(48, 24)
        self.up4 = up(24, 24)
        self.outc = outconv(24, n_channels)

        self.cos_loss_fn = nn.CosineSimilarity()
        if self.gpu:
            self.cuda()

    def forward(self, x):
        if self.N_Z == 128:
            x = self.up0(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        x = x / torch.sqrt((x ** 2).sum((1, 2, 3), keepdim=True))
        return x

    def loss(self, pred, gt):
        B = gt.shape[0]
        gt_var = torch.FloatTensor(gt)
        if self.gpu:
            gt_var = gt_var.cuda()

        pred = pred / torch.sqrt((pred ** 2).sum((1, 2, 3), keepdim=True))
        gt_var = gt_var / torch.sqrt((gt_var ** 2).sum((1, 2, 3), keepdim=True))

        # negative of absolute value of cosine similarity as loss function
        l = neg_expcos = -torch.mean(torch.abs(self.cos_loss_fn(pred.view(B, -1), gt_var.view(B, -1))))
        # PP^T
        if self.lmbd != 0:
            pt = torch.transpose(pred.view(B, -1), 0, 1)
            ppt = torch.mm(pred.view(B, -1), pt)
            ppt_norm = torch.norm(ppt)
            l = neg_expcos + self.lmbd*ppt_norm
        return l

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv ** 2).sum((1, 2, 3), keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        ps_var = self.forward(rv_var)
        ps = ps_var.detach().cpu().numpy()

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp + 1))
        return ps

    def calc_rho(self, gt, inp):
        N = gt.shape[0]
        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv ** 2).sum((1, 2, 3), keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        x_dec = self.forward(rv_var)
        x_dec = x_dec.squeeze(0).detach().cpu().numpy()
        rho = (gt * x_dec).sum() / np.sqrt((gt ** 2).sum() * (x_dec ** 2).sum())
        return rho

    def project(self, latent_Z):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess

        ps = self.forward(latent_Z)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp + 1))
        return ps




