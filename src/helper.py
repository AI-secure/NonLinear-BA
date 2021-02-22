import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--gpu_id', type=int)
    # parser.add_argument('--train', action='store_true')
    parser.add_argument('--account', type=int, default=0)
    parser.add_argument('--help_attack', action='store_true')
    parser.add_argument('--n_epoch', type=int)
    args = parser.parse_args()

    if args.help_attack:
        for i in range(args.n_epoch):
            if i == 2:
                continue
            cmd = 'CUDA_VISIBLE_DEVICES=%d python3 main_pytorch_multi.py --use_gpu --TASK cifar10 --cifar10_img_size 224 --pgen DCGAN_finetune%d' %(args.gpu_id, i)
            print(cmd)
            os.system(cmd)
    else:
        if args.account == 0:
            for pgen in ['PCA9408']:
            # for pgen in ['VAE9408', 'naive']: # 'DCT9408',
                cmd = 'CUDA_VISIBLE_DEVICES=%d python3 main_api_attack.py --pgen %s --use_gpu --account 0' %(args.gpu_id, pgen)
                if args.mounted:
                    cmd += ' --mounted'
                print(cmd)
                os.system(cmd)
        elif args.account == 1:
            for pgen in ['AE9408', 'GAN9408', 'resize9408']:
                cmd = 'CUDA_VISIBLE_DEVICES=%d python3 main_api_attack.py --pgen %s --use_gpu --account 1' %(args.gpu_id, pgen)
                if args.mounted:
                    cmd += ' --mounted'
                print(cmd)
                os.system(cmd)

    # if args.train:
    #     # '--TASK celeba'
    #     task_args = ['--TASK cifar10 --cifar10_img_size 224 --N_Z 9408',
    #                  '--TASK mnist --mnist_img_size 224 --N_Z 9408',
    #                  # '--TASK imagenet --N_Z 9408',
    #                  # '--TASK mnist --N_Z 196',
    #                  # '--TASK cifar10 --N_Z 768'
    #                  ]
    #     gpu_id = args.gpu_id
    #     for task_arg in task_args:
    #         cmd = 'CUDA_VISIBLE_DEVICES=%d taskset --cpu-list 0-11 python3 train_pca_generator.py %s' %(gpu_id, task_arg)
    #         if args.mounted:
    #             cmd += ' --mounted'
    #         print(cmd)
    #         os.system(cmd)
    #
    # else:
    #     task_args = [
    #                  '--TASK celeba',
    #                  '--TASK cifar10 --cifar10_img_size 224 --N_Z 9408',
    #                  '--TASK mnist --mnist_img_size 224 --N_Z 9408',
    #                  # '--TASK imagenet --N_Z 9408',
    #                  # '--TASK mnist --N_Z 196',
    #                  # '--TASK cifar10 --N_Z 768'
    #                  ]
    #     gpu_id = args.gpu_id
    #     for task_arg in task_args:
    #         cmd = 'CUDA_VISIBLE_DEVICES=%d python3 main_pytorch_multi.py --use_gpu --pgen PCA9408 %s --N_Z 9408 --suffix imagenet_basis' %(gpu_id, task_arg)
    #         if args.mounted:
    #             cmd += ' --mounted'
    #         print(cmd)
    #         os.system(cmd)

    # for REF in ['naive', 'AE9408', 'resize', 'DCT']:
    #     for tail in ['', '_u', '_s', '_v']:
    #         cmd = 'mv BAPP_result/mnist_%s_internal_dim%s.npy BAPP_result/mnist_224_%s_internal_dim%s.npy' %(REF, tail, REF, tail)
    #         os.system(cmd)

    # root_dirs = ['./gen_models', ]
    # for root_dir in root_dirs:
    #     for file_name in os.listdir(root_dir):
    #         if file_name.startswith('mnist_224_z3136_gradient_vae_3136_generator'):
    #             cmd = 'mv %s/%s %s/%s%s'%(root_dir, file_name, root_dir, 'mnist_224', file_name[15:])
    #             os.system(cmd)
    #         elif file_name.startswith('mnist10_224_i0_p98_z3136_gradient_vae_3136_generator'):
    #             cmd = 'mv %s/%s %s/%s%s' %(root_dir, file_name, root_dir, 'mnist10_224_i0_p98', file_name[24:])
    #             os.system(cmd)

    # root_dir = '../class_models'
    # for file_name in os.listdir(root_dir):
    #