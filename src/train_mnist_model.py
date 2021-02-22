import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models import MNISTDNN
from utils import epoch_train, epoch_eval
from tqdm import tqdm

import argparse

import model_settings

GPU = True
BATCH_SIZE=32


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
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    mnist_img_size = args.mnist_img_size
    n_class = 10

    # if mnist_img_size == 28:
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,)),
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize((mnist_img_size, mnist_img_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,)),
    #     ])
    transform = model_settings.get_data_transformation("mnist", args)
    # model_file_name, _ = model_settings.get_model_file_name("mnist", args)
    model_file_name = model_settings.get_model_file_name("mnist", args)
    print(model_file_name)

    trainset = torchvision.datasets.MNIST(root='../raw_data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='../raw_data/', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    if args.do_train:
        print("Train %d-way MNIST models with img size %d" % (n_class, mnist_img_size))
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = MNISTDNN(model_type=MODEL_TYPE, gpu=GPU, n_class=n_class, img_size=mnist_img_size)
            if MODEL_TYPE == 'vgg16':
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with tqdm(range(10)) as pbar:
                for _ in pbar:
                    train_loss, train_acc = epoch_train(model, optimizer, trainloader)
                    test_loss, test_acc = epoch_eval(model, testloader)
                    # torch.save(model.state_dict(),
                    #            '../class_models/mnist%d_%d_%s.model' % (n_class, mnist_img_size, MODEL_TYPE))
                    torch.save(model.state_dict(),
                               '../class_models/%s_%s.model' % (model_file_name, MODEL_TYPE))
                    pbar.set_description("Train acc %.4f, Test acc %.4f" % (train_acc, test_acc))
    else:
        print("Test %d-way MNIST models" %(n_class))
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = MNISTDNN(model_type=MODEL_TYPE, gpu=GPU, n_class=n_class, img_size=mnist_img_size)
            model.load_state_dict(torch.load('../class_models/%s_%s.model'%(model_file_name, MODEL_TYPE)))
            test_loss, test_acc = epoch_eval(model, testloader)
            print("Model type %s, Test acc %.4f"%(MODEL_TYPE, test_acc))