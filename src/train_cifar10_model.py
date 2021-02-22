# https://github.com/xiaojunxu/BAPP-improve/blob/master/foolbox-based/train_cifar_model.py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models import CifarDNN
from utils import epoch_train, epoch_eval
from tqdm import tqdm
import argparse

import model_settings

GPU = True
BATCH_SIZE=32

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar10_img_size', type=int, default=32)
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    img_size = args.cifar10_img_size

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = model_settings.get_data_transformation("cifar10", args)
    # model_file_name, _ = model_settings.get_model_file_name("cifar10", args)
    model_file_name = model_settings.get_model_file_name("cifar10", args)
    print(model_file_name)

    trainset = torchvision.datasets.CIFAR10(root='../raw_data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    if args.do_train:
        print("Train CIFAR10 models")
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = CifarDNN(model_type=MODEL_TYPE, gpu=GPU, img_size=img_size)
            if MODEL_TYPE == 'vgg16':
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with tqdm(range(10)) as pbar:
                for _ in pbar:
                    train_loss, train_acc = epoch_train(model, optimizer, trainloader)
                    test_loss, test_acc = epoch_eval(model, testloader)
                    torch.save(model.state_dict(), '../class_models/%s_%s.model'%(model_file_name, MODEL_TYPE))
                    pbar.set_description("Train acc %.4f, Test acc %.4f"%(train_acc, test_acc))

    else:
        print("test cifar10 models")
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = CifarDNN(model_type=MODEL_TYPE, gpu=GPU, img_size=img_size)
            model.load_state_dict(torch.load('../class_models/%s_%s.model'%(model_file_name, MODEL_TYPE)))
            test_loss, test_acc = epoch_eval(model, testloader)
            print("Model %s, Test acc %.4f"%(MODEL_TYPE, test_acc))