import sys
import numpy as np
# import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from models import GANDiscriminator, GANGenerator
# from dcgan_origin import DCGAN_D, DCGAN_G
import matplotlib.pyplot as plt
import utils
import argparse
import model_settings
import constants


def epoch_train(REF, modelD, modelG, optimizerD, optimizerG, N_Z, BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None, mounted=False):

    if TASK == 'celeba':
        # celeba train: 5087, test: 624
        N_train = 5087
        if REF == 'rnd':
            N_train = 15625
    elif TASK == 'imagenet':
        # test: 625; train: 8750
        N_train = 8750
        if REF == 'rnd':
            N_train = 15625
    elif TASK == 'celebaid' or TASK == 'celeba2':
        N_train = 9
        if REF == 'rnd':
            N_train = 5*9
    elif TASK == 'dogcat2':
        N_train = 625
        if REF == 'rnd':
            N_train = 5*625

    # elif TASK == 'mnist_224':
    #     N_train = 1875
    #     if REF == 'rnd':
    #         N_train = 5*1875
    # elif TASK == 'cifar10_224':
    #     N_train = 1563
    #     if REF == 'rnd':
    #         N_train = 5*1563

    elif TASK.startswith('mnist'):
        N_train = 1875
        if REF == 'rnd':
            N_train = 5*1875
    elif TASK.startswith('cifar10'):
        N_train = 1563
        if REF == 'rnd':
            N_train = 5*1563

    else:
        print("Not implemented")
        assert 0

    data_path = '%s/%s_%s/train_batch_%d.npy'

    # N_used = 200
    N_used = N_train
    modelD.train()
    modelG.train()
    perm = np.random.permutation(N_used)

    tot_num = 0.0
    cum_Dreal = 0.0
    cum_Dfake = 0.0
    cum_G = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (constants.ROOT_DATA_PATH, TASK, REF, idx)) # shape: (32, 150528)

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

    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    GPU = True
    TASK = args.TASK
    if TASK == 'mnist' or TASK == 'cifar10':
        TASK = model_settings.get_model_file_name(TASK, args) #TODO mnist N_Z 3136
    model_file_name = TASK

    if TASK.startswith('mnist'):
        n_channels = 1
    else:
        n_channels = 3

    N_Z = args.N_Z

    modelD = GANDiscriminator(n_channels=n_channels, gpu=GPU)
    modelG = GANGenerator(n_z=N_Z, n_channels=n_channels, gpu=GPU)

    if args.pretrained:
        modelD.load_state_dict(torch.load('./gen_models/normal_%s_gradient_gan_%d_discriminator.model' % (TASK, N_Z)))
        modelG.load_state_dict(torch.load('./gen_models/normal_%s_gradient_gan_%d_generator.model' % (TASK, N_Z)))
        model_file_name = 'finetuned_' + model_file_name

    optimizerD = torch.optim.RMSprop(modelD.parameters(), lr=5e-5)
    optimizerG = torch.optim.RMSprop(modelG.parameters(), lr=5e-5)
    fixed_noise = torch.FloatTensor(10, N_Z).normal_()
    
    with open('./gen_results/gan_%s_%d_loss_curve.txt' %(model_file_name, N_Z), 'w') as outf:
        for _ in range(200):
            fake = modelG(fixed_noise)

            fig = plt.figure(figsize=(20, 8))
            # for i in range(10):
            #     plt.subplot(2, 5, i + 1)
            #     to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
            #     to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
            #     plt.imshow(to_plt)
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
            plt.savefig('./plots/%s_gradient_gan%d_eg-%d.pdf' % (model_file_name, N_Z, _))
            plt.close(fig)

            print(epoch_train(REF='rnd', modelD=modelD, modelG=modelG, optimizerD=optimizerD, optimizerG=optimizerG, N_Z=N_Z, outf=outf, mounted=args.mounted))
            torch.save(modelD.state_dict(), './gen_models/%s_gradient_gan_%d_discriminator.model' % (model_file_name, N_Z))
            torch.save(modelG.state_dict(), './gen_models/%s_gradient_gan_%d_generator.model' % (model_file_name, N_Z))

            torch.save(modelD.state_dict(), './gen_models/%s_gradient_gan_%d_%d_discriminator.model' % (model_file_name, N_Z, _))
            torch.save(modelG.state_dict(), './gen_models/%s_gradient_gan_%d_%d_generator.model' % (model_file_name, N_Z, _))

    fake = modelG(fixed_noise)
    fig = plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
        to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
        plt.imshow(to_plt)
    plt.savefig('./plots/%s_gradient_gan%d_eg-final.pdf' %(model_file_name, N_Z))