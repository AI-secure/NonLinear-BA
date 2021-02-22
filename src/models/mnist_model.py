import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MNISTDNN(nn.Module):
    def __init__(self, model_type, pretrained=True, gpu=False, img_size=28, n_class=10):
        super(MNISTDNN, self).__init__()
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
        self.output = nn.Linear(1000, n_class)
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.shape[0]
        x = x.expand(B, 3, self.img_size, self.img_size)
        sf = int(224/self.img_size)
        rm = 224 % self.img_size
        assert rm == 0
        if sf > 1:
            x = F.interpolate(x, scale_factor=sf)
        x = self.model(x)
        x = self.output(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)