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


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
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


class AEGenerator(nn.Module):
    def __init__(self, n_channels, preprocess=None, gpu=False, N_Z=9408):
        super(AEGenerator, self).__init__()
        self.gpu = gpu
        self.preprocess = preprocess
        self.N_Z = N_Z
        #self.ENC_SHAPE = (10, 14, 14)
        #C,H,W = self.ENC_SHAPE

        self.inc = inconv(n_channels, 24)
        self.down1 = down(24, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 48)
        self.down4 = down(48, 48)

        if self.N_Z == 128:
            self.down5 = nn.Sequential(
                nn.Conv2d(in_channels=48, out_channels=8, kernel_size=10, stride=2, padding=1),
                nn.LeakyReLU()
            )
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
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = nn.CosineSimilarity()
        #self.outc1 = outconv(64, n_channels)
        #self.outc2 = outconv(64, n_channels)
        if self.gpu:
            self.cuda()

    def encode(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        if self.N_Z == 128:
            x = self.down5(x)

        x_enc = x / torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        return x_enc
        # return x

    def decode(self, x_enc):
        if self.N_Z == 128:
            x_enc = self.up0(x_enc)

        x = self.up1(x_enc)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        x = x / torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        return x

    def forward(self, x):
        x = torch.FloatTensor(x)
        if self.gpu:
            x = x.cuda()

        x_norm = torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        x = x / x_norm

        x_enc = self.encode(x)
        x = self.decode(x_enc)

        x = x * x_norm
        return x_enc, x

    def loss(self, pred, gt):
        gt_var = torch.FloatTensor(gt)
        if self.gpu:
            gt_var = gt_var.cuda()

        pred = pred / torch.sqrt((pred**2).sum((1,2,3),keepdim=True))
        gt_var = gt_var / torch.sqrt((gt_var**2).sum((1,2,3),keepdim=True))
        return self.loss_fn(pred, gt_var)
        #return 1-self.loss_fn(pred, gt_var).mean()

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv**2).sum((1,2,3),keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        ps_var = self.decode(rv_var)
        ps = ps_var.detach().cpu().numpy()

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp):
        x_enc, x_dec = self.forward(gt[None])
        x_dec = x_dec.squeeze(0).detach().cpu().numpy()
        rho = (gt*x_dec).sum() / np.sqrt( (gt**2).sum() * (x_dec**2).sum() )
        return rho

    def project(self, latent_Z):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess

        ps = self.decode(latent_Z)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps
