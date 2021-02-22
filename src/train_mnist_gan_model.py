import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models import GANGenerator, GANDiscriminator
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse

import model_settings

GPU = True
BATCH_SIZE=32


def epoch_train(REF, modelD, modelG, optimizerD, optimizerG, N_Z, trainloader,
                BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None, mounted=False):
    modelD.train()
    modelG.train()

    tot_num = 0.0
    cum_Dreal = 0.0
    cum_Dfake = 0.0
    cum_G = 0.0

    for X, y in trainloader:
        X = X / np.sqrt((X ** 2).sum(1, keepdims=True))

        B = X.shape[0]
        assert B <= BATCH_SIZE
        X = np.reshape(X, (B, n_channels, 224, 224))

        # Train D
        errD_real = 0.0
        errD_fake = 0.0
        for _ in range(D_ITERS):
            for p in modelD.parameters():
                p.data.clamp_(-0.01, 0.01)
            optimizerD.zero_grad()

            # Loss with real
            l_real = modelD(X)

            # Loss with fake
            noise = torch.FloatTensor(B, N_Z).normal_(0, 1)
            # noise = noise.resize_(B,N_Z,1,1).cuda()#origin
            fake = modelG(noise).detach()
            l_fake = modelD(fake)

            l = l_real - l_fake
            l.backward()
            optimizerD.step()
            errD_fake += l_fake.item()
            errD_real += l_real.item()
        errD_real = errD_real / D_ITERS
        errD_fake = errD_fake / D_ITERS

        # Train G
        errG = 0.0
        for _ in range(G_ITERS):
            optimizerG.zero_grad()
            noise = torch.FloatTensor(B, N_Z).normal_(0, 1)
            # noise = noise.resize_(B,N_Z,1,1).cuda()#origin
            fake = modelG(noise)
            l_G = modelD(fake)
            l_G.backward()
            optimizerG.step()
            errG += l_G.item()
        errG = errG / D_ITERS

        # Log result and show
        cum_Dreal += errD_real * B
        cum_Dfake += errD_fake * B
        cum_G += errG * B
        tot_num += B
        if outf is not None:
            outf.write('%.6f %.6f %.6f\n' % (errD_real, errD_fake, errG))

    return cum_Dreal / tot_num, cum_Dfake / tot_num, cum_G / tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # args.img_size
    # 28: original image
    # 224: images upsampled with interpolation
    parser.add_argument('--mnist_img_size', type=int, default=28)

    # args.padding_size
    # 0: no padding
    # for simplicity, if input size [A*A], output size [B*B], then padding_size = (B-A)/2
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--N_Z', type=int, default=9408)
    args = parser.parse_args()

    mnist_img_size = args.mnist_img_size
    n_class = 10

    TASK = 'mnist'

    TASK = model_settings.get_model_file_name(TASK, args)

    transform = model_settings.get_data_transformation("mnist", args)
    model_file_name = model_settings.get_model_file_name("mnist", args)
    print(model_file_name)

    trainset = torchvision.datasets.MNIST(root='../raw_data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='../raw_data/', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    if TASK.startswith('mnist'):
        n_channels = 1
    else:
        n_channels = 3

    N_Z = args.N_Z

    modelD = GANDiscriminator(n_channels=n_channels, gpu=GPU)

    modelG = GANGenerator(n_z=N_Z, n_channels=n_channels, gpu=GPU)

    optimizerD = torch.optim.RMSprop(modelD.parameters(), lr=5e-5)
    optimizerG = torch.optim.RMSprop(modelG.parameters(), lr=5e-5)
    fixed_noise = torch.FloatTensor(10, N_Z).normal_()

    with open('./gen_results/original_gan_%s_%d_loss_curve.txt' % (TASK, N_Z), 'w') as outf:
        for _ in range(200):
            fake = modelG(fixed_noise)

            fig = plt.figure(figsize=(20, 8))
            if n_channels == 3:
                for i in range(10):
                    plt.subplot(2, 5, i + 1)
                    to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
                    to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
                    plt.imshow(to_plt)
            else:
                for i in range(10):
                    plt.subplot(2, 5, i + 1)
                    x = fake[i].detach().cpu().numpy().reshape(224, 224)
                    to_plt = (x - x.min()) / (x.max() - x.min())
                    plt.imshow(to_plt, cmap='gray')
            plt.savefig('./plots/original_%s_gradient_gan%d_eg-%d.pdf' % (TASK, N_Z, _))
            plt.close(fig)

            print("original %s gan, epoch %d" %(TASK, _), epoch_train(REF='rnd', modelD=modelD, modelG=modelG, optimizerD=optimizerD, optimizerG=optimizerG,
                              N_Z=N_Z, trainloader=trainloader, outf=outf, mounted=args.mounted))
            torch.save(modelD.state_dict(), './gen_models/original_%s_gradient_gan_%d_discriminator.model' % (TASK, N_Z))
            torch.save(modelG.state_dict(), './gen_models/original_%s_gradient_gan_%d_generator.model' % (TASK, N_Z))

    fake = modelG(fixed_noise)
    fig = plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
        to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
        plt.imshow(to_plt)
    plt.savefig('./plots/original_%s_gradient_gan%d_eg-final.pdf' % (TASK, N_Z))