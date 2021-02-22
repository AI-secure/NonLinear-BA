import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import constants


def get_model_file_name(TASK, args):
    if TASK == 'mnist':
        original_size = 28
        mnist_img_size = args.mnist_img_size
        mnist_padding_size = args.mnist_padding_size
        mnist_padding_first = args.mnist_padding_first
        n_class = 10

        if mnist_padding_size == 0:
            model_file_name = "%s_%d" %(TASK, mnist_img_size)
        else:
            interp_size = mnist_img_size - original_size - mnist_padding_size*2
            assert interp_size >= 0
            if mnist_padding_first:
                model_file_name = "%s%d_%d_p%d_i%d" %(TASK, n_class, mnist_img_size, mnist_padding_size, interp_size)
            else:
                model_file_name = "%s%d_%d_i%d_p%d" %(TASK, n_class, mnist_img_size, interp_size, mnist_padding_size)

        # if args.N_Z == 3136:
        #     output_file_name = model_file_name + '_z%d' %(args.N_Z)

    elif TASK == 'cifar10':
        original_size = 32
        cifar10_img_size = args.cifar10_img_size
        cifar10_padding_size = args.cifar10_padding_size
        cifar10_padding_first = args.cifar10_padding_first
        n_class = 10

        if cifar10_padding_size == 0:
            model_file_name = "%s_%d" %(TASK, cifar10_img_size)
        else:
            interp_size = cifar10_img_size - original_size - cifar10_padding_size*2
            assert interp_size >= 0
            if cifar10_padding_first:
                model_file_name = "%s_%d_p%d_i%d" % (TASK, cifar10_img_size, cifar10_padding_size, interp_size)
            else:
                model_file_name = "%s_%d_i%d_p%d" % (TASK, cifar10_img_size, interp_size, cifar10_padding_size)

    else:
        model_file_name = TASK

    # if output_file_name is None:
    #     return model_file_name, model_file_name
    # else:
    #     return model_file_name, output_file_name
    return model_file_name


def get_data_transformation(TASK, args):
    _mean, _std = constants.get_mean_std(TASK=TASK)

    if TASK == 'mnist':
        original_size = 28
        mnist_img_size = args.mnist_img_size
        mnist_padding_size = args.mnist_padding_size
        mnist_padding_first = args.mnist_padding_first

        if mnist_img_size == 28:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_mean, _std),
            ])
        else:
            if mnist_padding_size > 0:
                if mnist_padding_first:
                    transform = transforms.Compose([
                        transforms.Pad(padding=mnist_padding_size, fill=0, padding_mode='edge'),
                        transforms.Resize((mnist_img_size, mnist_img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(_mean, _std),
                    ])
                else:
                    after_interp_size = mnist_img_size - mnist_padding_size * 2
                    assert after_interp_size >= 0
                    transform = transforms.Compose([
                        transforms.Resize((after_interp_size, after_interp_size)),
                        transforms.Pad(padding=mnist_padding_size, fill=0, padding_mode='edge'),
                        transforms.ToTensor(),
                        transforms.Normalize(_mean, _std),
                    ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((mnist_img_size, mnist_img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(_mean, _std),
                ])

    elif TASK == 'cifar10':
        cifar10_img_size = args.cifar10_img_size
        cifar10_padding_size = args.cifar10_padding_size
        cifar10_padding_first = args.cifar10_padding_first

        if cifar10_img_size == 32:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_mean, _std),
            ])
        else:
            if cifar10_padding_size > 0:
                if cifar10_padding_first:
                    transform = transforms.Compose([
                        transforms.Pad(padding=cifar10_padding_size, fill=0, padding_mode='edge'),
                        transforms.Resize((cifar10_img_size, cifar10_img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(_mean, _std),
                    ])
                else:
                    after_interp_size = cifar10_img_size - cifar10_padding_size * 2
                    assert after_interp_size >= 0
                    transform = transforms.Compose([
                        transforms.Resize((after_interp_size, after_interp_size)),
                        transforms.Pad(padding=cifar10_padding_size, fill=0, padding_mode='edge'),
                        transforms.ToTensor(),
                        transforms.Normalize(_mean, _std),
                    ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((cifar10_img_size, cifar10_img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(_mean, _std),
                ])

    else:
        print("Not implemented yet")
        assert 0

    return transform


def get_data_transformation_without_normalization(TASK, args):

    if TASK == 'mnist':
        original_size = 28
        mnist_img_size = args.mnist_img_size
        mnist_padding_size = args.mnist_padding_size
        mnist_padding_first = args.mnist_padding_first

        if mnist_img_size == 28:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            if mnist_padding_size > 0:
                if mnist_padding_first:
                    transform = transforms.Compose([
                        transforms.Pad(padding=mnist_padding_size, fill=0, padding_mode='edge'),
                        transforms.Resize((mnist_img_size, mnist_img_size)),
                        transforms.ToTensor(),
                    ])
                else:
                    after_interp_size = mnist_img_size - mnist_padding_size * 2
                    assert after_interp_size >= 0
                    transform = transforms.Compose([
                        transforms.Resize((after_interp_size, after_interp_size)),
                        transforms.Pad(padding=mnist_padding_size, fill=0, padding_mode='edge'),
                        transforms.ToTensor(),
                    ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((mnist_img_size, mnist_img_size)),
                    transforms.ToTensor(),
                ])

    elif TASK == 'cifar10':
        cifar10_img_size = args.cifar10_img_size
        cifar10_padding_size = args.cifar10_padding_size
        cifar10_padding_first = args.cifar10_padding_first

        if cifar10_img_size == 32:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            if cifar10_padding_size > 0:
                if cifar10_padding_first:
                    transform = transforms.Compose([
                        transforms.Pad(padding=cifar10_padding_size, fill=0, padding_mode='edge'),
                        transforms.Resize((cifar10_img_size, cifar10_img_size)),
                        transforms.ToTensor(),
                    ])
                else:
                    after_interp_size = cifar10_img_size - cifar10_padding_size * 2
                    assert after_interp_size >= 0
                    transform = transforms.Compose([
                        transforms.Resize((after_interp_size, after_interp_size)),
                        transforms.Pad(padding=cifar10_padding_size, fill=0, padding_mode='edge'),
                        transforms.ToTensor(),
                    ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((cifar10_img_size, cifar10_img_size)),
                    transforms.ToTensor(),
                ])

    else:
        print("Not implemented yet")
        assert 0

    return transform