import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--gpu_id', type=int)
    args = parser.parse_args()

    tasks = [
             '--TASK mnist --mnist_img_size 224',
             '--TASK cifar10 --cifar10_img_size 224',
             '--TASK imagenet',
             '--TASK celeba',
             ]

    pgens = ['DCT9408', 'VAE9408', 'naive', 'AE9408', 'GAN9408', 'resize9408', 'PCA9408']

    for pgen in pgens:
        for i_t in range(len(tasks)):
            task = tasks[i_t]
            cmd = 'CUDA_VISIBLE_DEVICES=%d python3 nonlinear_analysis.py --use_gpu --do_svd %s --pgen %s' %(args.gpu_id, task, pgen)
            if args.mounted:
                cmd += ' --mounted'
            print(cmd)
            os.system(cmd)
