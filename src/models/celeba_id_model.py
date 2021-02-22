import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models


class CelebAIDDNN(nn.Module):
    def __init__(self, model_type, num_class, pretrained=True, gpu=False):
        super(CelebAIDDNN, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu
        self.num_class = num_class

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

        self.output = nn.Linear(1000, self.num_class)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = self.model(x)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


# def epoch_train(model, optimizer, dataloader):
#     model.train()
#
#     cum_loss = 0.0
#     cum_acc = 0.0
#     tot_num = 0.0
#     for X, y in dataloader:
#         B = X.size()[0]
#         if (B==1):
#             continue
#         pred = model(X)
#         loss = model.loss(pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         cum_loss += loss.item() * B
#         pred_c = pred.max(1)[1].cpu()
#         cum_acc += (pred_c.eq(y)).sum().item()
#         tot_num = tot_num + B
#
#     print(cum_loss / tot_num, cum_acc / tot_num)
#     return cum_loss / tot_num, cum_acc / tot_num
#
#
# def epoch_eval(model, dataloader):
#     model.eval()
#
#     cum_loss = 0.0
#     cum_acc = 0.0
#     tot_num = 0.0
#     for X, y in dataloader:
#         B = X.size()[0]
#         with torch.no_grad():
#             pred = model(X)
#             loss = model.loss(pred, y)
#
#         cum_loss += loss.item() * B
#         pred_c = pred.max(1)[1].cpu()
#         cum_acc += (pred_c.eq(y)).sum().item()
#         tot_num = tot_num + B
#
#     print(cum_loss / tot_num, cum_acc / tot_num)
#     return cum_loss / tot_num, cum_acc / tot_num