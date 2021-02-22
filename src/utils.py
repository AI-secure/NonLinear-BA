import numpy as np
import torch
import matplotlib.pyplot as plt

import model_settings
import torchvision


def plot_training_data(data_path, img_name, x_shape=(3, 224, 224)):
    X = np.load(data_path)
    X = np.reshape(X[:10], (10, *x_shape))
    fig = plt.figure(figsize=(20, 8))
    if len(x_shape) == 3:
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            to_plt = X[i].transpose(1, 2, 0)
            to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
            plt.imshow(to_plt)
    else:
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            x = X[i]
            to_plt = (x-x.min()) / (x.max() - x.min())
            plt.imshow(to_plt, cmap='gray')
    plt.savefig('./plots/train_gradient_%s.pdf' % (img_name))
    plt.close(fig)


# regularize gradient arrays
# celeba: shape: (32, 150528)
# standardize each sample on its own instead the whole set of gradients coming from one REF model
# use after loading the gradient npy file
def regularize(X):
    mean_ = np.mean(X, axis=1, keepdims=True)
    std_ = np.std(X, axis=1, keepdims=True)
    # while np.sum((std_ == 0).astype(int)):
    #     r_min = np.argmin(std_)
    #     X = np.delete(X, r_min, axis=0)
    #     mean_ = np.mean(X, axis=1, keepdims=True)
    #     std_ = np.std(X, axis=1, keepdims=True)
    X = (X - mean_) / std_

    max_ = np.max(X, axis=1, keepdims=True)
    min_ = np.min(X, axis=1, keepdims=True)
    X = (X - min_) / (max_ - min_) * 2 - 1
    return X


# remove all-same (inf) lines
def validate(X):
    # mean_ = np.mean(X, axis=1, keepdims=True)
    std_ = np.std(X, axis=1, keepdims=True)
    while np.sum((std_ == 0).astype(int)):
        r_min = np.argmin(std_)
        X = np.delete(X, r_min, axis=0)
        # mean_ = np.mean(X, axis=1, keepdims=True)
        std_ = np.std(X, axis=1, keepdims=True)
    # X = (X - mean_) / std_
    return X


# def normalize(X):
#     mean_ = np.mean(X, axis=1, keepdims=True)
#     std_ = np.std(X, axis=1, keepdims=True)
#     X = (X - mean_) / std_
#
#     max_ = np.max(X, axis=1, keepdims=True)
#     min_ = np.min(X, axis=1, keepdims=True)
#     X = (X - min_) / (max_ - min_) * 2 - 1
#     return X

def epoch_train(model, optimizer, dataloader):
    model.train()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        if (B==1):
            continue
        pred = model(X)
        loss = model.loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    print(cum_loss / tot_num, cum_acc / tot_num)
    return cum_loss / tot_num, cum_acc / tot_num


def epoch_eval(model, dataloader):
    model.eval()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        with torch.no_grad():
            pred = model(X)
            loss = model.loss(pred, y)

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    print(cum_loss / tot_num, cum_acc / tot_num)
    return cum_loss / tot_num, cum_acc / tot_num


def calc_cos_sim(x1, x2, dim=1):
    cos = (x1 * x2).sum(dim) / np.sqrt((x1 ** 2).sum(dim) * (x2 ** 2).sum(dim))
    return cos


def get_imgset(TASK, args, is_train=True, BATCH_SIZE=32):
    if TASK == 'mnist':
        # model_file_name = model_settings.get_model_file_name("mnist", args)
        transform = model_settings.get_data_transformation("mnist", args)
        imgset = torchvision.datasets.MNIST(root='../raw_data/', train=is_train, download=True, transform=transform)
        imgloader = torch.utils.data.DataLoader(imgset, batch_size=BATCH_SIZE, shuffle=False)

    elif TASK == 'cifar10':
        # model_file_name = model_settings.get_model_file_name("cifar10", args)
        transform = model_settings.get_data_transformation("cifar10", args)
        imgset = torchvision.datasets.CIFAR10(root='../raw_data/', train=is_train, download=True, transform=transform)
        imgloader = torch.utils.data.DataLoader(imgset, batch_size=BATCH_SIZE, shuffle=True)

    else:
        print("Not implemented yet")
        assert 0

    return imgloader


