import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models


class CelebADNN(nn.Module):
    def __init__(self, model_type, pretrained=True, gpu=False):
        super(CelebADNN, self).__init__()
        self.gpu = gpu
        if model_type == 'dense121':
            self.model = models.densenet121(pretrained=pretrained).eval()
        elif model_type == 'res18':
            self.model = models.resnet18(pretrained=pretrained).eval()
        elif model_type == 'res50':
            self.model = models.resnet50(pretrained=pretrained).eval()
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained).eval()
        elif model_type == 'googlenet':
            self.model = models.googlenet(pretrained=pretrained, aux_logits=False).eval()
        elif model_type == 'wideresnet':
            self.model = models.wide_resnet50_2(pretrained=pretrained).eval()
        else:
            raise NotImplementedError()
        self.output = nn.Linear(1000, 2)
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        # x = F.interpolate(x, scale_factor=7)
        x = self.model(x)
        x = self.output(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


# class CelebAResNet(nn.Module):
#     def __init__(self, num_class, pretrained=True, gpu=False):
#         super(CelebAResNet, self).__init__()
#         self.pretrained = pretrained
#         self.gpu = gpu
#         self.num_class = num_class
#
#         self.resnet = models.resnet18(pretrained=pretrained)
#         self.output = nn.Linear(1000, self.num_class)
#
#         if gpu:
#             self.cuda()
#
#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         x = self.resnet(x)
#         x = self.output(x)
#
#         return x
#
#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)





# if __name__ == '__main__':
#     gpu = True
#     num_class = 10
#     n_e = 500
#     batch_size = 128
#
#     root_dir = '../raw_data/celeba'
#     do_random_sample = False
#
#     # celeba_dataset.preprocess_data(root_dir, n_id=num_class, random_sample=do_random_sample)
#     # celeba_dataset.sort_imgs(root_dir)
#     # celeba_dataset.get_dataset(root_dir, n_id=num_class, random_sample=do_random_sample)
#
#     trainset = CelebADataset(root_dir=root_dir, is_train=True, transform=transform, preprocess=False,
#                              random_sample=do_random_sample, n_id=num_class)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
#     testset = CelebADataset(root_dir=root_dir, is_train=False, transform=transform, preprocess=False,
#                              random_sample=do_random_sample, n_id=num_class)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
#     #print (len(trainset), len(testset))
#     #assert 0
#
#     resmodel = CelebAResNet(num_class, pretrained=True, gpu=gpu)
#     # optimizer = torch.optim.SGD(resmodel.parameters(), lr=1e-3, momentum=0.9)
#     optimizer = torch.optim.Adam(resmodel.output.parameters(), lr=1e-4, weight_decay=1e-5)
#
#     for e in range(n_e):
#         print("Epoch %d" %(e))
#         epoch_train(model=resmodel, optimizer=optimizer, dataloader=trainloader)
#         print("Evaluate")
#         epoch_eval(model=resmodel, dataloader=testloader)
#         torch.save(resmodel.state_dict(), '../models/celeba.model')
#
