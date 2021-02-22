import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models


class NClassDNN(nn.Module):
    def __init__(self, model_type, pretrained=True, gpu=False, n_class=2):
        super(NClassDNN, self).__init__()
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
        self.output = nn.Linear(1000, n_class)
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


