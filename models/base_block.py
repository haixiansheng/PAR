import math

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits = nn.Sequential(
            # nn.Linear(2048, nattr),
            nn.Linear(512, nattr),   # ##### resnet18需要修改
            nn.BatchNorm1d(nattr)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avg_pool = nn.MaxPool2d((8, 6))

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature):
        feat = self.avg_pool(feature)
        print(feat.shape)
        feat = self.avg_pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)
        return x


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params

    def finetune_params(self):
        return self.backbone.parameters()

    def forward(self, x, label=None):
        feat_map = self.backbone(x)
        logits = self.classifier(feat_map)
        return logits


class module(nn.Module):

    def __init__(self, model):
        super(module, self).__init__()

        self.module = model

    def forward(self, x, label=None):
        logits = self.module(x)
        return logits
