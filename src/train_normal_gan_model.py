import sys
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from models import GANDiscriminator, GANGenerator
import matplotlib.pyplot as plt
import utils
import argparse
import model_settings
from datasets import CelebAAttributeDataset


# def epoch_train(REF, mean_, std_, modelD, modelG, optimizerD, optimizerG, N_Z, BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None):
def epoch_train(REF, modelD, modelG, optimizerD, optimizerG, N_Z, trainloader, BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None, mounted=False):
    modelD.train()
    modelG.train()

    tot_num = 0.0
    cum_Dreal = 0.0
    cum_Dfake = 0.0
    cum_G = 0.0
    with tqdm(trainloader) as pbar:
        for X, labels in pbar:
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
            pbar.set_description("REF: %s: Cur Dreal/Dfake/G err: %.4f/%.4f/%.4f; Avg: %.4f/%.4f/%.4f" % (
            REF, errD_real, errD_fake, errG, cum_Dreal / tot_num, cum_Dfake / tot_num, cum_G / tot_num))
            if outf is not None:
                outf.write('%.6f %.6f %.6f\n' % (errD_real, errD_fake, errG))

    return cum_Dreal / tot_num, cum_Dfake / tot_num, cum_G / tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--N_Z', type=int)
    parser.add_argument('--mounted', action='store_true')

    parser.add_argument('--mnist_img_size', type=int, default=28)
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--cifar10_img_size', type=int, default=32)
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')
    args = parser.parse_args()

    GPU = True
    TASK = args.TASK

    if TASK == 'mnist' or TASK == 'cifar10':
        model_file_name = model_settings.get_model_file_name(TASK, args) #TODO mnist N_Z 3136
    else:
        model_file_name = TASK

    if TASK.startswith('mnist'):
        n_channels = 1
    else:
        n_channels = 3

    N_Z = args.N_Z
    BATCH_SIZE = 32

    modelD = GANDiscriminator(n_channels=n_channels, gpu=GPU)
    modelG = GANGenerator(n_z=N_Z, n_channels=n_channels, gpu=GPU)

    optimizerD = torch.optim.RMSprop(modelD.parameters(), lr=5e-5)
    optimizerG = torch.optim.RMSprop(modelG.parameters(), lr=5e-5)
    fixed_noise = torch.FloatTensor(10, N_Z).normal_()

    transform = model_settings.get_data_transformation(TASK, args)

    if args.TASK.startswith('cifar10'):
        trainset = torchvision.datasets.CIFAR10(root='../raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=True, transform=transform)
    elif args.TASK.startswith('mnist'):
        trainset = torchvision.datasets.MNIST(root='../raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='../raw_data/', train=False, download=True, transform=transform)
    # elif args.TASK == 'celeba':
    #     if mounted:
    #         root_dir = '/home/hcli/data/celebA'
    #     else:
    #         root_dir = '/data/hcli/celebA'
    #     attr_o_i = 'Mouth_Slightly_Open'
    #     list_attr_path = '%s/list_attr_celeba.txt' % (root_dir)
    #     img_data_path = '%s/img_align_celeba' % (root_dir)
    #     trainset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='train',
    #                                       transform=transform)
    #     testset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='test',
    #                                      transform=transform)
    elif args.TASK == 'imagenet':
        print("TODO")
        assert 0
    else:
        print("not implemented")
        assert 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    with open('./gen_results/normal_gan_%s_%d_loss_curve.txt' %(model_file_name, N_Z), 'w') as outf:
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
            plt.savefig('./plots/normal_%s_gradient_gan%d_eg-%d.pdf' % (model_file_name, N_Z, _))
            plt.close(fig)

            print(epoch_train(REF='rnd', modelD=modelD, modelG=modelG, optimizerD=optimizerD, optimizerG=optimizerG,
                              N_Z=N_Z, trainloader=trainloader, outf=outf, mounted=args.mounted))
            torch.save(modelD.state_dict(), './gen_models/normal_%s_gradient_gan_%d_discriminator.model' % (model_file_name, N_Z))
            torch.save(modelG.state_dict(), './gen_models/normal_%s_gradient_gan_%d_generator.model' % (model_file_name, N_Z))

            torch.save(modelD.state_dict(), './gen_models/normal_%s_gradient_gan_%d_%d_discriminator.model' % (model_file_name, N_Z, _))
            torch.save(modelG.state_dict(), './gen_models/normal_%s_gradient_gan_%d_%d_generator.model' % (model_file_name, N_Z, _))

    fake = modelG(fixed_noise)
    fig = plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
        to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
        plt.imshow(to_plt)
    plt.savefig('./plots/normal_%s_gradient_gan%d_eg-final.pdf' %(model_file_name, N_Z))