import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
import math


def plot_loss(TASK, N_Z):
    if TASK == 'celeba':
        N_train = 5087*5
    elif TASK == 'imagenet':
        N_train = 8750*5
    elif TASK.startswith('mnist'):
        N_train = 1875*5
    else:
        print("Not implemented")
        assert 0

    errD_reals = []
    errD_fakes = []
    errGs = []
    with open('./gen_results/gan_%s_%d_loss_curve.txt' %(TASK, N_Z), 'r') as inf:
        for line in inf:
            errD_real, errD_fake, errG = line.strip().split()
            errD_reals.append(float(errD_real))
            errD_fakes.append(float(errD_fake))
            errGs.append(float(errG))
    N_tot = len(errGs)
    n_errD_reals = []
    n_errD_fakes = []
    n_errGs = []
    for i in range(math.ceil(N_tot/N_train)):
        n_errD_reals.append(np.mean(errD_reals[i*N_train:(i+1)*N_train]))
        n_errD_fakes.append(np.mean(errD_fakes[i*N_train:(i+1)*N_train]))
        n_errGs.append(np.mean(errGs[i*N_train:(i+1)*N_train]))
    print(n_errD_reals)
    print(n_errD_fakes)
    print(n_errGs)
    xs = list(range(len(n_errGs)))
    fig = plt.figure(figsize=(20, 10))
    # plt.scatter(xs[:10000], errD_reals[:10000], label='errD_real')
    # plt.scatter(xs[:10000], errD_fakes[:10000], label='errD_fake')
    # plt.scatter(xs[:10000], errGs[:10000], label='errG')
    plt.plot(xs, n_errD_reals, label='errD_real', color='r')
    plt.plot(xs, n_errD_fakes, label='errD_fake', color='g')
    plt.plot(xs, n_errGs, label='errG', color='b')
    plt.legend()
    # plt.yscale('log')
    plt.xlabel('# Epochs')
    plt.ylabel('WGAN loss')
    plt.savefig('./plots/gan_loss_%s_%d.png' % (TASK, N_Z), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--TASK', type=str)
    # parser.add_argument('--N_Z', type=int)
    # args = parser.parse_args()
    #
    # plot_loss(TASK=args.TASK, N_Z=args.N_Z)

    # TASKs = ['celeba', 'imagenet']
    # N_Zs = [128, 9408]
    # for TASK in TASKs:
    #     for N_Z in N_Zs:
    #         print(TASK, N_Z)
    #         plot_loss(TASK=TASK, N_Z=N_Z)
    # TASK = 'mnist_224'
    TASK = 'mnist10_224_i0_p98'
    plot_loss(TASK=TASK, N_Z=9408)
