import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CifarDNN(nn.Module):
    def __init__(self, model_type, pretrained=True, gpu=False, img_size=32):
        super(CifarDNN, self).__init__()
        self.gpu = gpu
        self.img_size = img_size

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
        self.output = nn.Linear(1000, 10)
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        sf = int(224 / self.img_size)
        rm = 224 % self.img_size
        assert rm == 0
        if sf > 1:
            x = F.interpolate(x, scale_factor=sf)
        # x = F.interpolate(x, scale_factor=7)
        x = self.model(x)
        x = self.output(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class CifarResNet(nn.Module):
    def __init__(self, pretrained=True, gpu=False):
        super(CifarResNet, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu

        self.resnet = models.resnet18(pretrained=pretrained)
        self.output = nn.Linear(1000, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        #x = F.interpolate(x, [224,224])
        #x = self.resnet(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)

        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class CifarDenseNet(nn.Module):
    def __init__(self, pretrained=True, gpu=False):
        super(CifarDenseNet, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu

        self.densenet = models.densenet121(pretrained=pretrained)
        self.output = nn.Linear(1000, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        #x = F.interpolate(x, [224,224])
        #x = self.resnet(x)

        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.densenet.classifier(x)

        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

