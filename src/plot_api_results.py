import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib


def get_pairs():
    # get pairs
    pairs = []
    root_dir = './api_results'
    for filename in os.listdir(root_dir):
        if filename.startswith('api_attack_facepp_DCT9408'):
            src_tgt_str = filename[26:-4]
            src_tgt_lst = src_tgt_str.split('_')
            pairs.append([int(_id) for _id in src_tgt_lst])
    print(pairs)
    np.save('./api_results/src_tgt_pairs.npy', pairs)
    new_pairs = np.load('./api_results/src_tgt_pairs.npy')
    print(new_pairs)


def plot_mean_std(pgen, l_log=21):
    # mean and std
    logs = []
    root_dir = './api_results'
    for filename in os.listdir(root_dir):
        if filename.startswith('api_attack_facepp_%s'%(pgen)):
            with open('%s/%s' % (root_dir, filename)) as inf:
                l = list(json.load(inf))
                if len(l) != l_log:
                    print('%s %s log length abnormal'%(pgen, filename))
                    continue
                logs.append(l)
    logs = np.array(logs)

    avg_X = []
    avg_y = []
    std_y = []
    for t in range(logs.shape[1]):
        avg_X.append(np.mean(logs[:, t, 0]))
        avg_y.append(np.mean(logs[:, t, 1]))
        std_y.append(np.std(logs[:, t, 1]))
    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(avg_X, avg_y, yerr=std_y)
    plt.xlabel('# Queries')
    plt.ylabel('L2 Distance from Target Image')
    fig.savefig('./plots/api_facepp_%s_mean_distance.png' %(pgen), bbox_inches='tight')


def compare():
    # compare different methods
    # pairs = np.load('./api_results/src_tgt_pairs.npy')
    pairs = [[48144, 162607], [188917, 93663], [26603, 77495], [163922, 80037], [21260, 97657], [290, 1369]]
    colors = {'VAE9408': '#F34A17', 'DCT9408': '#0000CC'}
    root_dir = './api_results'
    for src_id, tgt_id in pairs:
        fig = plt.figure(figsize=(8, 6))
        for filename in os.listdir(root_dir):
            if filename.startswith('api') and filename.endswith('%d_%d.log' %(src_id, tgt_id)):
                with open('%s/%s' % (root_dir, filename)) as inf:
                    log = json.load(inf)
                    log = np.array(log)
                    # print(log[:, 0])
                    # print(log[:, 1])
                    plt.plot(log[:, 0], log[:, 1], label=filename[11:25], color=colors[filename[18:25]])
        plt.xlabel('# Queries')
        plt.ylabel('L2 Distance from Target Image')
        plt.title('src: %d; tgt: %d' %(src_id, tgt_id))
        plt.legend()
        fig.savefig('./plots/compare_%d_%d.png' %(src_id, tgt_id), bbox_inches='tight')


def plot_means(pgens, names, styles, rgb_colors, l_log=21):
    # mean and std
    fig = plt.figure(figsize=(10, 6))
    root_dir = './api_results'
    for i in range(len(pgens)):
        pgen = pgens[i]
        logs = []
        for filename in os.listdir(root_dir):
            if filename.startswith('api_attack_facepp_%s'%(pgen)):
                with open('%s/%s' % (root_dir, filename)) as inf:
                    l = list(json.load(inf))
                    if len(l) != l_log:
                        print('%s %s log length abnormal'%(pgen, filename))
                        continue
                    logs.append(l)
        logs = np.array(logs)

        avg_X = []
        avg_y = []
        for t in range(logs.shape[1]):
            avg_X.append(np.mean(logs[:, t, 0]))
            avg_y.append(np.mean(logs[:, t, 1]))
        plt.plot(avg_X, avg_y, label=names[i], linestyle=styles[i], color=rgb_colors[i])
    plt.xlabel('# Queries')
    plt.ylabel('L2 Distance from Target Image')
    anchor_loc = (1, 1)
    plt.legend(bbox_to_anchor=anchor_loc)
    fig.savefig('./plots/api_facepp_means.png', bbox_inches='tight')


if __name__ == '__main__':
    font = {'size': 20}
    matplotlib.rc('font', **font)
    # compare()
    pipeline_name = 'NLBA'
    pgens = ['naive', 'resize9408', 'DCT9408', 'PCA9408', 'AE9408', 'VAE9408', 'GAN9408']
    names = ['HSJA', 'QEBA-S', 'QEBA-F', 'QEBA-I', '%s-AE' %(pipeline_name), '%s-VAE' %(pipeline_name), '%s-GAN' %(pipeline_name)]
    styles = ['--', '-.', '-.', '-.', '-', '-', '-']
    rgb_colors = ['k', '#FF0000', '#FF8000', '#00994C', '#0066CC', '#8c564b', '#e377c2', '#bcbd22']
    # for pgen in pgens:
    #     plot_mean_std(pgen=pgen, l_log=21)

    plot_means(pgens, names, styles, rgb_colors, l_log=21)
