import os
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from models import CifarResNet, CifarDNN
from datasets import ImageNetDataset
from datasets import CelebAAttributeDataset
from datasets import CelebAIDDataset
from models import CelebADNN
from models import CelebAIDDNN
import argparse

from datasets import DogCatDataset
from models import NClassDNN
from models import MNISTDNN

from datasets import CelebABinIDDataset

import model_settings
import constants


def calc_gt_grad(ref_model, Xs, preprocess_std):
    X_withg = torch.autograd.Variable(Xs, requires_grad=True)
    #score = ref_model(X_withg).max(1)[0].mean()
    #score.backward()
    scores = ref_model(X_withg)
    labs = scores.max(1)[1]
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = loss_fn(scores, labs)
    loss.backward()
    grad = X_withg.grad.data
    grad = grad / torch.FloatTensor(np.array(preprocess_std)[:,None,None]).cuda()
    return grad


def load_model_dataset(TASK, REF, root_dir):
    if TASK == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        trainset = ImageNetDataset(train=True, transform=transform)
        testset = ImageNetDataset(train=False, transform=transform)
        if REF == 'dense121':
            ref_model = models.densenet121(pretrained=True).eval()
        elif REF == 'res18':
            ref_model = models.resnet18(pretrained=True).eval()
        elif REF == 'res50':
            ref_model = models.resnet50(pretrained=True).eval()
        elif REF == 'vgg16':
            ref_model = models.vgg16(pretrained=True).eval()
        elif REF == 'googlenet':
            ref_model = models.googlenet(pretrained=True).eval()
        elif REF == 'wideresnet':
            ref_model = models.wide_resnet50_2(pretrained=True).eval()
        if GPU:
            ref_model.cuda()
        preprocess_std = (0.229, 0.224, 0.225)

    elif TASK == 'celeba':
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        attr_o_i = 'Mouth_Slightly_Open'
        list_attr_path = '%s/celebA/list_attr_celeba.txt' % (root_dir)
        img_data_path = '%s/celebA/img_align_celeba' % (root_dir)
        trainset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='train',
                                          transform=transform)
        testset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='test',
                                         transform=transform)

        ref_model = CelebADNN(model_type=REF, pretrained=False, gpu=GPU)
        ref_model.load_state_dict(torch.load('../class_models/celeba_%s_%s.model' % (attr_o_i, REF)))
        preprocess_std = (0.5, 0.5, 0.5)

    elif TASK == 'celebaid':
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        num_class = 10
        trainset = CelebAIDDataset(root_dir=root_dir, is_train=True, transform=transform, preprocess=False,
                                   random_sample=False, n_id=num_class)
        testset = CelebAIDDataset(root_dir=root_dir, is_train=False, transform=transform, preprocess=False,
                                  random_sample=False, n_id=num_class)

        ref_model = CelebAIDDNN(model_type=REF, num_class=num_class, pretrained=False, gpu=GPU).eval()
        ref_model.load_state_dict(torch.load('../class_models/celeba_id_%s.model' % (REF)))

        preprocess_std = (0.5, 0.5, 0.5)

    elif TASK == 'dogcat2':
        BATCH_SIZE = 32
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        num_class = 2
        trainset = DogCatDataset(root_dir=f'{constants.ROOT_DATA_PATH}/dogcat', mode='train', transform=transform)
        testset = DogCatDataset(root_dir=f'{constants.ROOT_DATA_PATH}/dogcat', mode='test', transform=transform)

        ref_model = NClassDNN(model_type=REF, pretrained=False, gpu=GPU, n_class=num_class)
        ref_model.load_state_dict(torch.load('../class_models/%s_%s.model' % ('dogcat2', REF)))

        preprocess_std = (1, 1, 1)

    elif TASK == 'cifar10':
        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        mean, std = constants.plot_mean_std(TASK)
        transform = model_settings.get_data_transformation(TASK, args)
        model_file_name = model_settings.get_model_file_name(TASK, args)

        trainset = torchvision.datasets.CIFAR10(root='%s/'%(constants.ROOT_DATA_PATH), train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='%s/'%(constants.ROOT_DATA_PATH), train=False, download=True, transform=transform)

        ref_model = CifarDNN(model_type=REF, gpu=GPU, pretrained=False, img_size=cifar10_img_size)
        ref_model.load_state_dict(torch.load('../class_models/%s_%s.model' % (model_file_name, REF)))
        if GPU:
            ref_model.cuda()
        preprocess_std = std

    elif TASK == 'mnist':
        mean, std = constants.plot_mean_std(TASK)
        transform = model_settings.get_data_transformation(TASK, args)
        model_file_name = model_settings.get_model_file_name(TASK, args)
        num_class = 10
        trainset = torchvision.datasets.MNIST(root='%s/'%(constants.ROOT_DATA_PATH), train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='%s/'%(constants.ROOT_DATA_PATH), train=False, download=True, transform=transform)

        ref_model = MNISTDNN(model_type=REF, gpu=GPU, n_class=num_class, img_size=mnist_img_size)
        ref_model.load_state_dict(torch.load('../class_models/%s_%s.model' % (model_file_name, REF)))

        preprocess_std = std

    elif TASK == 'celeba2':
        image_size = 224
        num_class = 2
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        root_dir = f"{constants.ROOT_DATA_PATH}"
        trainset = CelebABinIDDataset(poi=args.celeba_poi, root_dir=root_dir, is_train=True, transform=transform,
                                      get_data=False, random_sample=False, n_id=10)
        testset = CelebABinIDDataset(poi=args.celeba_poi, root_dir=root_dir, is_train=False, transform=transform,
                                     get_data=False, random_sample=False, n_id=10)

        ref_model = NClassDNN(model_type=REF, pretrained=False, gpu=GPU, n_class=num_class)
        ref_model.load_state_dict(torch.load('../class_models/%s_%s.model' % ('celeba2', REF)))

        preprocess_std = (0.5, 0.5, 0.5)

    else:
        assert 0

    return ref_model, trainset, testset, preprocess_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--mounted', action='store_true')

    parser.add_argument('--celeba_poi', type=int, default=0)

    parser.add_argument('--mnist_img_size', type=int, default=28)
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--cifar10_img_size', type=int, default=32)
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')

    args = parser.parse_args()

    root_dir = f"{constants.ROOT_DATA_PATH}"

    GPU = True
    N_used = 999999
    TASK = args.TASK
    BATCH_SIZE = 32

    mnist_img_size = args.mnist_img_size
    cifar10_img_size = args.cifar10_img_size


    for REF in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
        print ("Task: %s; Ref model: %s"%(TASK, REF))

        ref_model, trainset, testset, preprocess_std = load_model_dataset(TASK=TASK, REF=REF, root_dir=root_dir)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

        if TASK == 'mnist' or TASK == 'cifar10':
            model_file_name = model_settings.get_model_file_name(TASK, args)
            path = '%s/%s_%s' %(root_dir, model_file_name, REF)
            print(path)
        else:
            path = '%s/%s_%s'%(root_dir, TASK, REF)
        if not os.path.isdir(path):
            os.mkdir(path)
        i = 0
        for Xs, _ in tqdm(trainloader):
            if GPU:
                Xs = Xs.cuda()
            grad_gt = calc_gt_grad(ref_model, Xs, preprocess_std=preprocess_std) # shape: [3, 224, 224]
            grad_gt = grad_gt.reshape(grad_gt.shape[0], -1)
            np.save(path+'/train_batch_%d.npy'%i, grad_gt.cpu().numpy())
            i += 1
            if (i * BATCH_SIZE >= N_used):
                break

        i = 0
        for Xs, _ in tqdm(testloader):
            if GPU:
                Xs = Xs.cuda()
            grad_gt = calc_gt_grad(ref_model, Xs, preprocess_std=preprocess_std)
            grad_gt = grad_gt.reshape(grad_gt.shape[0], -1)
            np.save(path+'/test_batch_%d.npy'%i, grad_gt.cpu().numpy())
            i += 1
            if (i * BATCH_SIZE >= N_used):
                break

