import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
# import seaborn as sns
new_pipeline_name = 'NLBA'


def plot_logs(path, lc, label, ATTR_ID, ls='-'):
    with open(path) as inf:
        logs = json.load(inf)
    print (len(logs))
    for i, log in enumerate(logs):
        organized_data = zip(*log)
        Xs = [entry[0] for entry in log]
        ys = [entry[1+ATTR_ID] for entry in log]
        if i == 0:
            plt.plot(Xs, ys, linecolor=lc, linestype=ls, label=label)
        else:
            plt.plot(Xs, ys, linecolor=lc, linestyle=ls)

def mean(l, logarithm=False):
    if logarithm:
        l_lg = [np.log(x) for x in l]
        return np.exp(sum(l_lg) / len(l_lg))
    else:
        return sum(l) / len(l)

def plot_fig(TASK, rgb_colors):
    if TASK == 'imagenet':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_imagenet_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_imagenet_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
            ('BAPP_result/attack_multi_imagenet_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
            ('BAPP_result/attack_multi_imagenet_PCA9408_.log', rgb_colors[2], 'QEBA-I', '-.'),
            ('BAPP_result/attack_multi_imagenet_AE9408_.log', rgb_colors[3], '%s-AE' %(new_pipeline_name)),
            ('BAPP_result/attack_multi_imagenet_VAE9408_.log', rgb_colors[4], '%s-VAE' %(new_pipeline_name)),
            ('BAPP_result/attack_multi_imagenet_GAN9408_.log', rgb_colors[5], '%s-GAN' %(new_pipeline_name)),
        ]
    elif TASK == 'imagenet_compare':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_imagenet_naive_.log', 'k', 'HSJA', '--'),
            # ('BAPP_result/attack_multi_imagenet_resize9408_.log', rgb_colors[0], 'QEBA-S', '--'),
            ('BAPP_result/attack_multi_imagenet_DCT9408_.log', rgb_colors[1], 'QEBA-F9408', '--'),
            # ('BAPP_result/attack_multi_imagenet_PCA9408basis_.log', 'r-', 'QEBA-I'),
            # ('BAPP_result/attack_multi_imagenet_AE128_.log', rgb_colors[3], 'QEBA-AE128'),
            ('BAPP_result/attack_multi_imagenet_AE9408_.log', rgb_colors[2], 'QEBA-AE9408', '--'),

            ('BAPP_result/attack_multi_imagenet_DCT2352_.log', rgb_colors[3], 'QEBA-F2352'),
            ('BAPP_result/attack_multi_imagenet_DCT4107_.log', rgb_colors[6], 'QEBA-F4107'),
            ('BAPP_result/attack_multi_imagenet_DCT16428_.log', rgb_colors[5], 'QEBA-F16428'),
        ]
    elif TASK == 'celeba':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_celeba_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_celeba_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
            ('BAPP_result/attack_multi_celeba_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
            # ('BAPP_result/attack_multi_celeba_PCA9408_.log', rgb_colors[2], 'QEBA-I9408', '--'),
            ('BAPP_result/attack_multi_celeba_PCA9408_imagenet_basis.log', rgb_colors[2], 'QEBA-I', '-.'),
            # ('BAPP_result/attack_multi_celeba_AE128_.log', rgb_colors[3], 'QEBA-AE128'),
            ('BAPP_result/attack_multi_celeba_AE9408_.log', rgb_colors[3], '%s-AE' %(new_pipeline_name)),
            # ('BAPP_result/attack_multi_celeba_oldAE9408_.log', rgb_colors[5], 'QEBA-oldAE9408'),
            ('BAPP_result/attack_multi_celeba_VAE9408_.log', rgb_colors[4], '%s-VAE' %(new_pipeline_name)),
            # ('BAPP_result/attack_multi_celeba_GAN128_.log', rgb_colors[8], 'QEBA-GAN128'),
            ('BAPP_result/attack_multi_celeba_GAN9408_.log', rgb_colors[5], '%s-GAN' %(new_pipeline_name)),
            # ('BAPP_result/attack_multi_celeba_var_.log', 'k-', 'QEBA-var'),
            # ('BAPP_result/attack_multi_celeba_expcos9408_.log', rgb_colors[9], 'QEBA-expcos9408', '-.'),
        ]
    elif TASK == 'celeba_compare':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_celeba_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_celeba_resize9408_.log', rgb_colors[0], 'QEBA-S9408', '--'),
            ('BAPP_result/attack_multi_celeba_DCT9408_.log', rgb_colors[1], 'QEBA-F9408', '--'),
            ('BAPP_result/attack_multi_celeba_AE9408_.log', rgb_colors[4], 'QEBA-AE9408', '--'),
            ('BAPP_result/attack_multi_celeba_VAE9408_.log', rgb_colors[6], 'QEBA-VAE9408', '--'),
            ('BAPP_result/attack_multi_celeba_GAN9408_.log', rgb_colors[7], 'QEBA-GAN9408', '--'),

            # ('BAPP_result/attack_multi_celeba_naive_d05.log', 'k', 'HSJA-0.5delta'),
            # ('BAPP_result/attack_multi_celeba_resize9408_d05.log', rgb_colors[0], 'QEBA-S9408-0.5delta'),
            # ('BAPP_result/attack_multi_celeba_DCT9408_d05.log', rgb_colors[1], 'QEBA-F9408-0.5delta'),
            # ('BAPP_result/attack_multi_celeba_AE9408_d05.log', rgb_colors[4], 'QEBA-AE9408-0.5delta'),
            # ('BAPP_result/attack_multi_celeba_VAE9408_d05.log', rgb_colors[6], 'QEBA-VAE9408-0.5delta'),
            # ('BAPP_result/attack_multi_celeba_GAN9408_d05.log', rgb_colors[7], 'QEBA-GAN9408-0.5delta'),
            #
            # ('BAPP_result/attack_multi_celeba_naive_d01.log', 'k', 'HSJA-0.1delta', '-.'),
            # ('BAPP_result/attack_multi_celeba_resize9408_d01.log', rgb_colors[0], 'QEBA-S9408-0.1delta', '-.'),
            # ('BAPP_result/attack_multi_celeba_DCT9408_d01.log', rgb_colors[1], 'QEBA-F9408-0.1delta', '-.'),
            # ('BAPP_result/attack_multi_celeba_AE9408_d01.log', rgb_colors[4], 'QEBA-AE9408-0.1delta', '-.'),
            # ('BAPP_result/attack_multi_celeba_VAE9408_d01.log', rgb_colors[6], 'QEBA-VAE9408-0.1delta', '-.'),
            # ('BAPP_result/attack_multi_celeba_GAN9408_d01.log', rgb_colors[7], 'QEBA-GAN9408-0.1delta', '-.'),

            ('BAPP_result/attack_multi_celeba_VAE9408_smooth0005.log', rgb_colors[6], 'QEBA-VAE9408-smooth0005', '-.'),
        ]
    elif TASK == 'celebaid':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_celebaid_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_celebaid_resize9408_.log', rgb_colors[0], 'QEBA-S9408', '--'),
            ('BAPP_result/attack_multi_celebaid_DCT9408_.log', rgb_colors[1], 'QEBA-F9408', '--'),
            # ('BAPP_result/attack_multi_celebaid_PCA9408_.log', rgb_colors[2], 'QEBA-I9408'),
            # ('BAPP_result/attack_multi_celebaid_AE128_.log', rgb_colors[3], 'QEBA-AE128'),
            ('BAPP_result/attack_multi_celebaid_AE9408_.log', rgb_colors[4], 'QEBA-AE9408'),
            ('BAPP_result/attack_multi_celebaid_VAE9408_.log', rgb_colors[6], 'QEBA-VAE9408'),
            # ('BAPP_result/attack_multi_celebaid_GAN128_.log', rgb_colors[8], 'QEBA-GAN128'),
            ('BAPP_result/attack_multi_celebaid_GAN9408_.log', rgb_colors[7], 'QEBA-GAN9408'),
        ]
    elif TASK == 'cifar10':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_cifar10_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_cifar10_resize192_.log', rgb_colors[0], 'QEBA-S192', '--'),
            ('BAPP_result/attack_multi_cifar10_DCT192_.log', rgb_colors[1], 'QEBA-F192', '--'),
            # ('BAPP_result/attack_multi_cifar10_PCA192_.log', rgb_colors[2], 'QEBA-I192'),
            ('BAPP_result/attack_multi_cifar10_AE192_.log', rgb_colors[4], 'QEBA-AE192'),
            # ('BAPP_result/attack_multi_cifar10_VAE192_.log', rgb_colors[6], 'QEBA-VAE192'),
            # ('BAPP_result/attack_multi_cifar10_GAN192_.log', rgb_colors[7], 'QEBA-GAN192'),
        ]
    elif TASK == 'cifar10_224':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_cifar10_224_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_cifar10_224_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
            ('BAPP_result/attack_multi_cifar10_224_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
            ('BAPP_result/attack_multi_cifar10_224_PCA9408_.log', rgb_colors[2], 'QEBA-I', '-.'),
            # ('BAPP_result/attack_multi_cifar10_224_PCA9408_imagenet_basis.log', rgb_colors[2], 'QEBA-I9408', '--'),
            ('BAPP_result/attack_multi_cifar10_224_AE9408_.log', rgb_colors[3], '%s-AE' %(new_pipeline_name)),
            ('BAPP_result/attack_multi_cifar10_224_VAE9408_.log', rgb_colors[4], '%s-VAE' %(new_pipeline_name)),
            # ('BAPP_result/attack_multi_cifar10_224_GAN9408_.log', rgb_colors[7], 'QEBA-GAN9408'),
            ('BAPP_result/attack_multi_cifar10_224_DCGAN_.log', rgb_colors[5], '%s-GAN' %(new_pipeline_name)),
            # ('BAPP_result/attack_multi_cifar10_224_DCGAN_finetune2_.log', rgb_colors[0], 'QEBA-DCGAN-finetune2')
        ]
        # for i in range(8):
        #     PLOT_INFO.append(('BAPP_result/attack_multi_cifar10_224_DCGAN_finetune%d_.log'%(i), rgb_colors[int(i%8)], 'QEBA-DCGAN-finetune-%d'%(i)))
    elif TASK == 'cifar10_224_compare':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_cifar10_224_naive_.log', 'k', 'HSJA', '--'),
            # ('BAPP_result/attack_multi_cifar10_224_resize9408_.log', rgb_colors[0], 'QEBA-S9408', '--'),
            ('BAPP_result/attack_multi_cifar10_224_DCT9408_.log', rgb_colors[1], 'QEBA-F9408', '--'),
            # # ('BAPP_result/attack_multi_cifar10_224_PCA9408_.log', rgb_colors[2], 'QEBA-I192'),
            # ('BAPP_result/attack_multi_cifar10_224_AE9408_.log', rgb_colors[4], 'QEBA-AE9408'),
            ('BAPP_result/attack_multi_cifar10_224_VAE9408_.log', rgb_colors[6], 'QEBA-VAE9408'),
            # ('BAPP_result/attack_multi_cifar10_224_GAN9408_.log', rgb_colors[7], 'QEBA-GAN9408'),

            # ('BAPP_result/attack_multi_cifar10_224_VAE9408_smooth1en8.log', rgb_colors[3], 'QEBA-VAE9408-smooth1e-8',
            #  '-.'),
            # ('BAPP_result/attack_multi_cifar10_224_VAE9408_smooth5en8_1.log', rgb_colors[4], 'QEBA-VAE9408-smooth5e-8_1',
            #  '-.'),
            # ('BAPP_result/attack_multi_cifar10_224_VAE9408_smooth1en7.log', rgb_colors[5], 'QEBA-VAE9408-smooth1e-7',
            #  '-.'),
            # ('BAPP_result/attack_multi_cifar10_224_VAE9408_smooth1en6.log', rgb_colors[6], 'QEBA-VAE9408-smooth1e-6',
            #  '-.'),
            # ('BAPP_result/attack_multi_cifar10_224_VAE9408_smooth0005.log', rgb_colors[1], 'QEBA-VAE9408-smooth5e-4',
            #  '-.'),
        ]
    elif TASK == 'cifar10_224_i0_p96':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_%s_naive_.log' %(TASK), 'k', 'HSJA', '--'),
            # ('BAPP_result/attack_multi_%s_resize9408_.log' %(TASK), rgb_colors[0], 'QEBA-S9408', '--'),
            ('BAPP_result/attack_multi_%s_DCT9408_.log' %(TASK), rgb_colors[1], 'QEBA-F9408', '--'),
            # ('BAPP_result/attack_multi_%s_PCA9408_.log' %(TASK), rgb_colors[2], 'QEBA-I192'),
            ('BAPP_result/attack_multi_%s_AE9408_.log' %(TASK), rgb_colors[4], 'QEBA-AE9408'),
            ('BAPP_result/attack_multi_%s_VAE9408_.log' %(TASK), rgb_colors[6], 'QEBA-VAE9408'),
            ('BAPP_result/attack_multi_%s_GAN9408_.log' %(TASK), rgb_colors[7], 'QEBA-GAN9408'),
        ]
    elif TASK == 'mnist':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_mnist_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_mnist_resize_.log', rgb_colors[0], 'QEBA-S', '--'),
            ('BAPP_result/attack_multi_mnist_DCT_.log', rgb_colors[1], 'QEBA-F', '--'),
            # ('BAPP_result/attack_multi_mnist_PCA12_.log', rgb_colors[2], 'QEBA-I12'),
            ('BAPP_result/attack_multi_mnist_AE12_.log', rgb_colors[4], 'QEBA-AE12'),
            ('BAPP_result/attack_multi_mnist_AE196_.log', rgb_colors[8], 'QEBA-AE196'),
            # ('BAPP_result/attack_multi_mnist_VAE12_.log', rgb_colors[6], 'QEBA-VAE12'),
            # ('BAPP_result/attack_multi_mnist_GAN12_.log', rgb_colors[7], 'QEBA-GAN12'),
        ]
    elif TASK == 'mnist_224':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_%s_naive_.log' %(TASK), 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_%s_resize9408_.log' %(TASK), rgb_colors[0], 'QEBA-S', '-.'),
            ('BAPP_result/attack_multi_%s_DCT9408_.log' %(TASK), rgb_colors[1], 'QEBA-F', '-.'),
            ('BAPP_result/attack_multi_%s_PCA9408_.log' %(TASK), rgb_colors[2], 'QEBA-I', '-.'),
            ('BAPP_result/attack_multi_%s_AE9408_.log' %(TASK), rgb_colors[3], '%s-AE' %(new_pipeline_name)),
            ('BAPP_result/attack_multi_%s_VAE9408_.log' %(TASK), rgb_colors[4], '%s-VAE' %(new_pipeline_name)),
            # ('BAPP_result/attack_multi_%s_GAN9408_.log' %(TASK), rgb_colors[7], 'QEBA-GAN9408'),
            ('BAPP_result/attack_multi_%s_DCGAN_.log' % (TASK), rgb_colors[5], '%s-GAN' %(new_pipeline_name)),
        ]
    elif TASK == 'mnist_224_compare':
        task = 'mnist_224'
        PLOT_INFO = [
            ('BAPP_result/attack_multi_%s_naive_.log' %(task), 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_%s_resize9408_.log' %(task), rgb_colors[0], 'QEBA-S 3136', '--'),
            ('BAPP_result/attack_multi_%s_DCT9408_.log' %(task), rgb_colors[1], 'QEBA-F 3136', '--'),
            # ('BAPP_result/attack_multi_%s_PCA12_.log' %(task), rgb_colors[2], 'QEBA-I12'),
            ('BAPP_result/attack_multi_%s_AE9408_.log' %(task), rgb_colors[4], 'QEBA-AE9408'),
            ('BAPP_result/attack_multi_%s_VAE9408_.log' %(task), rgb_colors[6], 'QEBA-VAE9408'),
            ('BAPP_result/attack_multi_%s_GAN9408_.log' %(task), rgb_colors[7], 'QEBA-GAN9408'),

            ('BAPP_result/attack_multi_%s_VAE9408_smooth0005.log' % (task), rgb_colors[6], 'QEBA-VAE9408-smooth0005', '-.'),
        ]
    elif TASK == 'mnist10_224_i0_p98':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_%s_naive_.log' %(TASK), 'k', 'HSJA', '--'),
            # ('BAPP_result/attack_multi_%s_resize9408_.log' %(TASK), rgb_colors[0], 'QEBA-S 3136', '--'),
            ('BAPP_result/attack_multi_%s_DCT9408_.log' %(TASK), rgb_colors[1], 'QEBA-F 3136', '--'),
            # ('BAPP_result/attack_multi_%s_PCA12_.log' %(TASK), rgb_colors[2], 'QEBA-I12'),
            ('BAPP_result/attack_multi_%s_AE9408_.log' %(TASK), rgb_colors[4], 'QEBA-AE9408'),
            ('BAPP_result/attack_multi_%s_VAE9408_.log' %(TASK), rgb_colors[6], 'QEBA-VAE9408'),
            ('BAPP_result/attack_multi_%s_z3136_VAE3136_.log' %(TASK), rgb_colors[8], 'QEBA-VAE3136'),
            ('BAPP_result/attack_multi_%s_GAN9408_.log' %(TASK), rgb_colors[7], 'QEBA-GAN9408'),
        ]
    elif TASK == 'dogcat2':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_dogcat2_naive_.log', 'k', 'HSJA', '--'),
            ('BAPP_result/attack_multi_dogcat2_resize9408_.log', rgb_colors[0], 'QEBA-S', '--'),
            ('BAPP_result/attack_multi_dogcat2_DCT9408_.log', rgb_colors[1], 'QEBA-F', '--'),
            # ('BAPP_result/attack_multi_dogcat2_PCA9408_.log', rgb_colors[2], 'QEBA-I9408'),
            ('BAPP_result/attack_multi_dogcat2_AE9408_.log', rgb_colors[4], 'QEBA-AE9408'),
            ('BAPP_result/attack_multi_dogcat2_VAE9408_.log', rgb_colors[6], 'QEBA-VAE9408'),
            ('BAPP_result/attack_multi_dogcat2_GAN9408_.log', rgb_colors[7], 'QEBA-GAN9408'),
        ]
    ATTR_NAME = {0:'dist', 1:'cos-est_vs_gt', 2:'rho', 3:'cos-est_vs_dist', 4: 'cos-gt_vs_dist'}

    all_logs = []
    for info in PLOT_INFO:
        with open(info[0]) as inf:
            all_logs.append(json.load(inf))

    # logs_arr = np.array(all_logs[0])
    # print(logs_arr.shape)
    # assert 0

    ### Plot error bar
    # for logs, info in zip(all_logs, PLOT_INFO):
    #     fig = plt.figure(figsize=(3, 3))
    #     avg_X = []
    #     avg_y = []
    #     std_y = []
    #     for t in range(len(logs[0])):
    #         avg_X.append(mean([log[t][0] for log in logs]))
    #         avg_y.append(mean([log[t][1] for log in logs]))
    #         std_y.append(np.std([log[t][1] for log in logs]))
    #     plt.errorbar(avg_X, avg_y, yerr=std_y)
    #     plt.legend()
    #     if TASK == 'cifar':
    #         plt.xlim([0, 5000])
    #     else:
    #         plt.xlim([0, 20000])
    #         plt.xticks([0, 5000, 10000, 15000, 20000], ['0', '5K', '10K', '15K', '20K'])
    #     plt.xlabel('# Queries')
    #     plt.ylabel('Mean Square Error')
    #     plt.title('%s-%s'%(TASK, info[2]))
    #     plt.savefig('./plots/multi_%s_errorbar%s.png' %(TASK, info[2]), bbox_inches='tight')
    # assert 0

    # fig_size = (8, 8)
    # anchor_loc = (1, 1)

    fig_size = (6, 6)

    ### Plot mean img
    fig = plt.figure(figsize=fig_size)
    for logs, info in zip(all_logs, PLOT_INFO):
        avg_X = []
        avg_y = []
        for t in range(len(logs[0])):
            avg_X.append(mean([log[t][0] for log in logs]))
            avg_y.append(mean([log[t][1] for log in logs], logarithm=True))
            #avg_y.append(mean([log[t][1] for log in logs], logarithm=False))
        # print (avg_X)
        # print (avg_y)
        if len(info) > 3:
            plt.plot(avg_X, avg_y, info[1], linestyle=info[3], label=info[2])
        else:
            plt.plot(avg_X, avg_y, info[1], label=info[2])
    anchor_loc = (1, 0.3)
    plt.legend(bbox_to_anchor=anchor_loc)
    plt.yscale('log')
    if TASK == 'cifar':
        plt.xlim([0,5000])
    else:
        plt.xlim([0,20000])
        plt.xticks([0,5000,10000,15000,20000],['0','5K','10K','15K','20K'])
    plt.xlabel('# Queries')
    plt.ylabel('L2 Distance from Target Image')
    plt.savefig('./plots/multi_%s_mean.png'%TASK, bbox_inches='tight')
    plt.close(fig)
    # assert 0

    ### Plot mean img zoom in
    fig = plt.figure(figsize=fig_size)
    for logs, info in zip(all_logs, PLOT_INFO):
        avg_X = []
        avg_y = []
        for t in range(len(logs[0])):
            avg_X.append(mean([log[t][0] for log in logs]))
            avg_y.append(mean([log[t][1] for log in logs], logarithm=True))
        n_zoom = int(len(avg_X)*0.2)
        if len(info) > 3:
            plt.plot(avg_X[:n_zoom], avg_y[:n_zoom], info[1], linestyle=info[3], label=info[2])
        else:
            plt.plot(avg_X[:n_zoom], avg_y[:n_zoom], info[1], label=info[2])
    anchor_loc = (1, 1)
    plt.legend(bbox_to_anchor=anchor_loc)
    plt.yscale('log')
    plt.xlabel('# Queries')
    plt.ylabel('L2 Distance from Target Image')
    plt.savefig('./plots/multi_%s_mean_zoom.png' % TASK, bbox_inches='tight')
    plt.close(fig)

    ### Plot mean stat
    # stat_id = 2 rho
    for stat_id in [2]:
        fig = plt.figure(figsize=fig_size)
        for logs, info in zip(all_logs, PLOT_INFO):
            #print (logs[0])
            avg_X = []
            avg_y = []
            for t in range(len(logs[0])):
                avg_X.append(mean([log[t][0] for log in logs]))
                avg_y.append(mean([log[t][stat_id+1] for log in logs]))
                #avg_y.append(mean([log[t][1] for log in logs], logarithm=False))
            # print (info[2], avg_X, avg_y)
            if len(info) > 3:
                plt.plot(avg_X, avg_y, info[1], linestyle=info[3], label=info[2])
            else:
                plt.plot(avg_X, avg_y, info[1], label=info[2])
            # print(info[2], avg_y)
        anchor_loc = (1, 0.55)
        plt.legend(bbox_to_anchor=anchor_loc)
        if stat_id == 0:
            plt.yscale('log')
        if TASK == 'cifar':
            plt.xlim([0,5000])
        else:
            plt.xlim([0,20000])
            plt.xticks([0,5000,10000,15000,20000],['0','5K','10K','15K','20K'])
        plt.xlabel('# Queries')
        plt.ylabel(ATTR_NAME[stat_id])
        plt.savefig('./plots/multi_%s_%s.png'%(TASK,ATTR_NAME[stat_id]), bbox_inches='tight')
        plt.close(fig)


    # plot omega value
    fig = plt.figure(figsize=fig_size)
    for logs, info in zip(all_logs, PLOT_INFO):
        avg_X = []
        avg_y = []
        for t in range(len(logs[0])):
            avg_X.append(mean([log[t][0] for log in logs]))
            avg_y.append(mean([log[t][6] for log in logs], logarithm=False))

        n_zoom = int(len(avg_X) * 0.2)
        if len(info) > 3:
            plt.plot(avg_X[:n_zoom], avg_y[:n_zoom], info[1], linestyle=info[3], label=info[2])
        else:
            plt.plot(avg_X[:n_zoom], avg_y[:n_zoom], info[1], label=info[2])
    anchor_loc = (1.7, 0.8)
    plt.legend(bbox_to_anchor=anchor_loc)
    plt.xlabel('# Queries')
    plt.ylabel('Omega')
    plt.savefig('./plots/multi_%s_omega.png' % TASK, bbox_inches='tight')
    plt.close(fig)

    # plot cos sim value
    fig = plt.figure(figsize=fig_size)
    for logs, info in zip(all_logs, PLOT_INFO):
        avg_X = []
        avg_y = []
        for t in range(len(logs[0])):
            avg_X.append(mean([log[t][0] for log in logs]))
            avg_y.append(mean([log[t][2] for log in logs], logarithm=False))

        if len(info) > 3:
            plt.plot(avg_X, avg_y, info[1], linestyle=info[3], label=info[2])
        else:
            plt.plot(avg_X, avg_y, info[1], label=info[2])
    anchor_loc = (1, 0.8)
    plt.legend(bbox_to_anchor=anchor_loc)
    plt.xlabel('# Queries')
    plt.ylabel('Cos sim')
    plt.savefig('./plots/multi_%s_cos_sim.png' % TASK, bbox_inches='tight')
    plt.close(fig)

    ### Plot success rate
    if TASK == 'imagenet':
        thresh = 1e-3
    elif TASK == 'celeba':
        thresh = 1e-4
    elif TASK == 'mnist_224':
        thresh = 5e-3
    elif TASK == 'cifar10_224':
        thresh = 1e-4
    else:
        print("Not implemented")
        assert 0
    fig = plt.figure(figsize=fig_size)
    for logs, info in zip(all_logs, PLOT_INFO):
        all_success = []
        for log in logs:
            qs = [entry[0] for entry in log]
            ds = [entry[1] for entry in log]
            success_nq = None
            for q, d in zip(qs, ds):
                if (d < thresh):
                    success_nq = q
                    break
            if success_nq is None:
                success_nq =499999
            all_success.append(success_nq)
        if len(info) > 3:
            plt.plot(sorted(all_success)+[999999], list(np.arange(len(all_success)) / len(all_success))+[1.0], info[1], linestyle=info[3], label=info[2])
        else:
            plt.plot(sorted(all_success)+[999999], list(np.arange(len(all_success)) / len(all_success))+[1.0], info[1], label=info[2])
    anchor_loc = (1, 0.8)
    plt.legend(bbox_to_anchor=anchor_loc)
    plt.xlim([0,20000])
    plt.xticks([0,5000,10000,15000,20000],['0','5K','10K','15K','20K'])
    plt.ylim([0,1])
    plt.xlabel('# Queries')
    plt.ylabel('Success rate at %.0e'%thresh)
    plt.savefig('./plots/multi_%s_success.pdf'%TASK, bbox_inches='tight')
    plt.close(fig)

    ### success rate table
    #THRESH = [1e-2,1e-3,1e-4]
    #NQUERY = [5000,10000,20000]
    #print ('\\begin{table*}[h]')
    #print ('    \\centering')
    #print ('    \\begin{tabular}{|'+'c|'*(1+len(THRESH))+'}')
    #print ('\t\\hline')
    #print ('\t%s/%s/%s/%s & %.2f & %.3f & %.4f \\\\'%(tuple(info[2] for info in PLOT_INFO)+tuple(THRESH)))
    #print ('\t\\hline')
    #for nq in NQUERY:
    #    print ('\t%d & '%nq,end='')
    #    first1 = True
    #    for dist in THRESH:
    #        if first1:
    #            first1 = False
    #        else:
    #            print (' & ', end='')
    #        first2 = True
    #        for logs, info in zip(all_logs, PLOT_INFO):
    #            if first2:
    #                first2=False
    #            else:
    #                print (' / ', end='')
    #            all_success = []
    #            for log in logs:
    #                qs = [entry[0] for entry in log]
    #                ds = [entry[1] for entry in log]
    #                for q, d in zip(qs, ds):
    #                    if (q > nq):
    #                        all_success.append(0)
    #                        break
    #                    elif (d < dist):
    #                        all_success.append(1)
    #                        break
    #            val = mean(all_success)
    #            print ('%.2f'%val, end='')
    #    print (' \\\\')
    #print ('\t\\hline')
    #print ('    \\end{tabular}')
    #print ('    \\caption{%s}'%( TASK ))
    #print ('    \\label{tab:result-%s}'%TASK)
    #print ('\\end{table*}')
    #assert 0
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_success = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (q > nq):
    #                all_success.append(0)
    #                break
    #            elif (d < dist):
    #                all_success.append(1)
    #                break
    #    val = mean(all_success)
    #    print ("%s, dist %e, #q %d, success@(#q,dist) = %.2f"%(info[2], dist, nq, val))

    ### Plot some img
    #ATTR_IDs = [0,1,2]
    #N_img = 10
    #N_repeat = 1
    #img_st = 40
    #fig = plt.figure(figsize=(N_img*5,len(ATTR_IDs)*5))
    #for img_id in range(N_img):
    #    for n_attr, attr_id in enumerate(ATTR_IDs):
    #        #plt.subplot(N_img, len(ATTR_IDs), img_id*len(ATTR_IDs)+n_attr+1)
    #        plt.subplot(len(ATTR_IDs), N_img, n_attr*N_img+img_id+1)
    #        for log, info in zip(all_logs, PLOT_INFO):
    #            plt.plot((1000,1000),(1e-3,1e-3), info[1], label=info[2])
    #            for log_id in range((img_st+img_id)*N_repeat, (img_st+img_id+1)*N_repeat):
    #                Xs = [entry[0] for entry in log[log_id]]
    #                ys = [entry[1+attr_id] for entry in log[log_id]]
    #                plt.plot(Xs, ys, info[1])
    #        plt.title(ATTR_NAME[attr_id])
    #        if attr_id == 0:
    #            plt.yscale('log')
    #        plt.legend()
    #fig.savefig('multi_%s_%s-%s.pdf'%(TASK, img_st, img_st+N_img-1))
    #assert 0

    ### Calc success@(#q,dist)
    #dist = 1e-4
    #nq = 10000
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_success = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (q > nq):
    #                all_success.append(0)
    #                break
    #            elif (d < dist):
    #                all_success.append(1)
    #                break
    #    val = mean(all_success)
    #    print ("%s, dist %e, #q %d, success@(#q,dist) = %.4f"%(info[2], dist, nq, val))


    ### Calc #q@dist
    #dist=1e-5
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_nq = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (d < dist):
    #                break
    #        all_nq.append(q)
    #    val = mean(all_nq, logarithm=True)
    #    print ("%s, dist %e, #q@dist = %.1f"%(info[2], dist, val))

    ### Calc dist@#q
    #nq = 10000
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_dist = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (q > nq):
    #                break
    #        all_dist.append(d)
    #    val = mean(all_dist, logarithm=True)
    #    print ("%s, #q %d, dist@#q = %e"%(info[2], nq, val))

def plot_n_perline(rgb_colors):
    info_dict = {1: 'mean', 2: 'cos_sim', 6: 'omega'}
    i_info = 6
    fig_size = (24, 5)
    fig = plt.figure(figsize=fig_size)
    n = len(TASKs)
    for i in range(n):
        TASK = TASKs[i]
        if TASK == 'imagenet':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_imagenet_naive_.log', 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_imagenet_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_imagenet_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_imagenet_PCA9408_.log', rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_imagenet_AE9408_.log', rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_imagenet_VAE9408_.log', rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_imagenet_GAN9408_.log', rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        elif TASK == 'celeba':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_celeba_naive_.log', 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_celeba_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_celeba_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_celeba_PCA9408_imagenet_basis.log', rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_celeba_AE9408_.log', rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_celeba_VAE9408_.log', rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_celeba_GAN9408_.log', rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        elif TASK == 'cifar10_224':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_cifar10_224_naive_.log', 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_cifar10_224_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_cifar10_224_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_cifar10_224_PCA9408_.log', rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_cifar10_224_AE9408_.log', rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_cifar10_224_VAE9408_.log', rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_cifar10_224_DCGAN_.log', rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        elif TASK == 'mnist_224':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_%s_naive_.log' % (TASK), 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_%s_resize9408_.log' % (TASK), rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_%s_DCT9408_.log' % (TASK), rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_%s_PCA9408_.log' % (TASK), rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_%s_AE9408_.log' % (TASK), rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_%s_VAE9408_.log' % (TASK), rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_%s_DCGAN_.log' % (TASK), rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        else:
            assert 0

        all_logs = []
        for info in PLOT_INFO:
            with open(info[0]) as inf:
                all_logs.append(json.load(inf))

        plt.subplot(1, n, i+1)
        for logs, info in zip(all_logs, PLOT_INFO):
            avg_X = []
            avg_y = []
            for t in range(len(logs[0])):
                avg_X.append(mean([log[t][0] for log in logs]))
                if i_info == 1:
                    avg_y.append(mean([log[t][i_info] for log in logs], logarithm=True))
                else:
                    avg_y.append(mean([log[t][i_info] for log in logs]))
            if len(info) > 3:
                plt.plot(avg_X, avg_y, info[1], linestyle=info[3], label=info[2])
            else:
                plt.plot(avg_X, avg_y, info[1], label=info[2])
        if i_info == 1:
            plt.yscale('log')
        plt.xticks([0, 5000, 10000, 15000, 20000], ['0', '5K', '10K', '15K', '20K'])
        if i == 0:
            if i_info == 1:
                plt.ylabel('L2 Distance from Target Image')
            elif i_info == 2:
                plt.ylabel('Cosine Similarity')
            elif i_info == 6:
                plt.ylabel('Omega')
        plt.xlabel('%s # Queries' %(TASK_names[i]))
    anchor_loc = (1, 1.0)
    plt.legend(bbox_to_anchor=anchor_loc)
    # plt.legend()
    # plt.show()
    plt.savefig('./plots/multi_n_perline_%s.png' %(info_dict[i_info]), bbox_inches='tight')
    plt.close(fig)


def plot_attack_success_rate_n_perline(rgb_colors):
    fig_size = (24, 5)
    fig = plt.figure(figsize=fig_size)
    n = len(TASKs)
    for i in range(n):
        TASK = TASKs[i]
        if TASK == 'imagenet':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_imagenet_naive_.log', 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_imagenet_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_imagenet_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_imagenet_PCA9408_.log', rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_imagenet_AE9408_.log', rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_imagenet_VAE9408_.log', rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_imagenet_GAN9408_.log', rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        elif TASK == 'celeba':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_celeba_naive_.log', 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_celeba_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_celeba_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_celeba_PCA9408_imagenet_basis.log', rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_celeba_AE9408_.log', rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_celeba_VAE9408_.log', rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_celeba_GAN9408_.log', rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        elif TASK == 'cifar10_224':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_cifar10_224_naive_.log', 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_cifar10_224_resize9408_.log', rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_cifar10_224_DCT9408_.log', rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_cifar10_224_PCA9408_.log', rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_cifar10_224_AE9408_.log', rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_cifar10_224_VAE9408_.log', rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_cifar10_224_DCGAN_.log', rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        elif TASK == 'mnist_224':
            PLOT_INFO = [
                ('BAPP_result/attack_multi_%s_naive_.log' % (TASK), 'k', 'HSJA', '--'),
                ('BAPP_result/attack_multi_%s_resize9408_.log' % (TASK), rgb_colors[0], 'QEBA-S', '-.'),
                ('BAPP_result/attack_multi_%s_DCT9408_.log' % (TASK), rgb_colors[1], 'QEBA-F', '-.'),
                ('BAPP_result/attack_multi_%s_PCA9408_.log' % (TASK), rgb_colors[2], 'QEBA-I', '-.'),
                ('BAPP_result/attack_multi_%s_AE9408_.log' % (TASK), rgb_colors[3], '%s-AE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_%s_VAE9408_.log' % (TASK), rgb_colors[4], '%s-VAE' % (new_pipeline_name)),
                ('BAPP_result/attack_multi_%s_DCGAN_.log' % (TASK), rgb_colors[5], '%s-GAN' % (new_pipeline_name)),
            ]
        else:
            assert 0

        if TASK == 'imagenet':
            thresh = 1e-3
        elif TASK == 'celeba':
            thresh = 1e-4
        elif TASK == 'mnist_224':
            thresh = 5e-3
        elif TASK == 'cifar10_224':
            thresh = 1e-4
        else:
            print("Not implemented")
            assert 0

        all_logs = []
        for info in PLOT_INFO:
            with open(info[0]) as inf:
                all_logs.append(json.load(inf))

        x_lim = 20000
        plt.subplot(1, n, i + 1)
        for logs, info in zip(all_logs, PLOT_INFO):
            all_success = []
            for log in logs:
                qs = [entry[0] for entry in log]
                ds = [entry[1] for entry in log]
                success_nq = None
                for q, d in zip(qs, ds):
                    if (d < thresh):
                        success_nq = q
                        break
                if success_nq is None:
                    success_nq = 499999
                all_success.append(success_nq)
            if len(info) > 3:
                plt.plot(sorted(all_success), list(np.arange(len(all_success)) / len(all_success)),
                         info[1], linestyle=info[3], label=info[2])
            else:
                plt.plot(sorted(all_success), list(np.arange(len(all_success)) / len(all_success)),
                         info[1], label=info[2])
        plt.xlim([0, 20000])
        plt.xticks([0, 5000, 10000, 15000, 20000], ['0', '5K', '10K', '15K', '20K'])
        plt.ylim([0, 1])
        if i == 0:
            plt.ylabel('Attack Success Rate')
        plt.xlabel('%s # Queries' % (TASK_names[i]))
    anchor_loc = (1, 1.0)
    # plt.legend(bbox_to_anchor=anchor_loc)
    plt.savefig('./plots/multi_n_perline_%s.png' % ('attack_success_rate'), bbox_inches='tight')
    plt.close(fig)
#     ### Plot success rate

#     fig = plt.figure(figsize=fig_size)

#     anchor_loc = (1, 0.8)
#     plt.legend(bbox_to_anchor=anchor_loc)
#     plt.xlim([0, 20000])
#     plt.xticks([0, 5000, 10000, 15000, 20000], ['0', '5K', '10K', '15K', '20K'])
#     plt.ylim([0, 1])
#     plt.xlabel('# Queries')
#     plt.ylabel('Success rate at %.0e' % thresh)
#     plt.savefig('./plots/multi_%s_success.pdf' % TASK, bbox_inches='tight')
#     plt.close(fig)


if __name__ == '__main__':
    font = {'size': 18}
    matplotlib.rc('font', **font)

    # sea_palette = sns.color_palette()
    # print(sea_palette)
    # rgb_colors = ['#%02x%02x%02x' % (int(sea_palette[i][0]*255), int(sea_palette[i][1]*255), int(sea_palette[i][2]*255)) for i in range(len(sea_palette))]
    # rgb_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # print(rgb_colors)
    # assert 0

    # TASK = 'celeba_compare'
    # TASK = 'cifar10_224_compare'
    # TASK = 'imagenet_compare'
    # TASK = 'mnist_224_compare'

    # TASK = 'celeba'
    # TASK = 'imagenet'
    # TASK = 'cifar10_224'
    # TASK = 'mnist_224'
    # TASK = 'mnist10_224_i0_p98'
    # TASK = 'cifar10_224_i0_p96'

    rgb_colors = ['#FF0000', '#FF8000', '#00994C', '#0066CC', '#8c564b', '#e377c2', '#bcbd22']
    TASKs = ['imagenet', 'celeba', 'cifar10_224', 'mnist_224']
    TASK_names = ['(a) ImageNet', '(b) CelebA', '(c) CIFAR10', '(d) MNIST']
    # plot_n_perline(rgb_colors)
    plot_attack_success_rate_n_perline(rgb_colors)
    # for TASK in TASKs:
    #     print(TASK)
    #     plot_fig(TASK=TASK, rgb_colors=rgb_colors)
    #     break