import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import model_settings
from models import Cifar10DCDiscriminator, Cifar10DCGenerator


def test_cifar10_dcgan(D=None, G=None, model_file_name='cifar10', suffix=''):
    if D is None or G is None:
        D = Cifar10DCDiscriminator(ngpu=1)
        G = Cifar10DCGenerator(ngpu=1)

        D.load_state_dict(torch.load('./models/weights/Cifar10_netD_epoch_199.pth'), strict=False)
        G.load_state_dict(torch.load('./models/weights/Cifar10_netG_epoch_199.pth'), strict=False)
        D = D.cuda()
        G = G.cuda()

    batch_size = 25
    latent_size = 100

    fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
    if torch.cuda.is_available():
        fixed_noise = fixed_noise.cuda()
    fake_images = G(fixed_noise)

    fake_images_np = fake_images.cpu().detach().numpy()
    fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, 224, 224)
    fake_images_np = fake_images_np.transpose((0, 2, 3, 1))
    R, C = 5, 5
    for i in range(batch_size):
        plt.subplot(R, C, i + 1)
        # plt.imshow(fake_images_np[i], interpolation='bilinear')
        to_plt = fake_images_np[i]
        max_ = np.max(to_plt, axis=1, keepdims=True)
        min_ = np.min(to_plt, axis=1, keepdims=True)
        to_plt = (to_plt - min_) / (max_ - min_) * 2 - 1
        # plt.imshow(fake_images_np[i])
        plt.imshow(to_plt)
    plt.savefig('./gen_results/%s_dcgan_test_%s.png'%(model_file_name, suffix))


if __name__ == '__main__':
    # test_cifar10_dcgan()
    # assert 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar10_img_size', type=int, default=32)  # resize cifar10
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')
    parser.add_argument('--mounted', action='store_true')
    args = parser.parse_args()
    args.TASK = 'cifar10'

    n_channels = 3

    netD = Cifar10DCDiscriminator(ngpu=1)
    netG = Cifar10DCGenerator(ngpu=1)

    netD.load_state_dict(torch.load('./models/weights/Cifar10_netD_epoch_199.pth'), strict=False)
    netG.load_state_dict(torch.load('./models/weights/Cifar10_netG_epoch_199.pth'), strict=False)
    netD = netD.cuda()
    netG = netG.cuda()

    lr = 1e-5
    beta1 = 0.5
    optimizerD = optim.Adam(netD.in_conv.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.out_conv.parameters(), lr=lr, betas=(beta1, 0.999))

    N_Z = 100
    BATCH_SIZE = 32
    fixed_noise = torch.randn(BATCH_SIZE, N_Z, 1, 1)
    fixed_noise = fixed_noise.cuda()

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()

    if args.mounted:
        data_path = 'ANONYMIZED_DIRECTORY/data/%s_%s/train_batch_%d.npy'
    else:
        data_path = 'ANONYMIZED_DIRECTORY/%s_%s/train_batch_%d.npy'
    REF = 'rnd'
    N_train = 5 * 1563
    netD.train()
    netG.train()
    perm = np.random.permutation(N_train)

    model_file_name = model_settings.get_model_file_name(TASK=args.TASK, args=args)
    with open('./gen_results/dcgan_%s_%d_loss_curve.txt' %(model_file_name, N_Z), 'w') as outf:
        for _e in range(200):
            with tqdm(perm) as pbar:
                for idx in pbar:
                    X = np.load(data_path % (model_file_name, REF, idx))  # shape: (32, 150528)
                    X = X / np.sqrt((X ** 2).sum(1, keepdims=True))

                    B = X.shape[0]
                    assert B <= BATCH_SIZE
                    X = np.reshape(X, (B, n_channels, 224, 224))

                    # https://github.com/csinva/gan-vae-pretrained-pytorch
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # train with real
                    netD.zero_grad()
                    if not torch.is_tensor(X):
                        X_tensor = torch.FloatTensor(X)
                    real_cpu = X_tensor.cuda()
                    batch_size = real_cpu.size(0)
                    label = torch.full((batch_size,), real_label)
                    label = label.cuda()

                    output = netD(real_cpu)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()

                    # train with fake
                    noise = torch.randn(batch_size, N_Z, 1, 1)
                    noise = noise.cuda()
                    fake = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake.detach())
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    errD = errD_real + errD_fake
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    output = netD(fake)
                    errG = criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    optimizerG.step()

                    pbar.set_description('Epoch %d, Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                                         %(_e, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # do checkpointing
            torch.save(netG.state_dict(), './gen_models/%s_netG_%d_epoch%d.pth' % (model_file_name, N_Z, _e))
            torch.save(netD.state_dict(), './gen_models/%s_netD_%d_epoch%d.pth' % (model_file_name, N_Z, _e))
            test_cifar10_dcgan(D=netD, G=netG, model_file_name=model_file_name, suffix='%d'%(_e))
