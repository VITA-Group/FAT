import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1.0)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


# Defines the new fc layer and classification layer
# |--Conv--|--bn--|--relu--|
class ClassBlock_Conv(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=True, num_bottleneck=256):
        super(ClassBlock_Conv, self).__init__()
        conv_block = []
        conv_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, padding=0, bias=False)]
        conv_block += [nn.BatchNorm2d(num_bottleneck)]

        if relu:
            # conv_block += [nn.LeakyReLU(0.1)]
            conv_block += [nn.ReLU()]
        if dropout:
            conv_block += [nn.Dropout(p=0.5)]
        conv_block = nn.Sequential(*conv_block)
        conv_block.apply(weights_init_kaiming)

        self.conv_block = conv_block

    def forward(self, x):
        x = self.conv_block(x)
        return x


# Defines the new fc layer and classification layer
# |--Linear--|
class ClassBlock_Linear(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=True, num_bottleneck=256):
        super(ClassBlock_Linear, self).__init__()

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]

        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

    def reset(self):
        self.classifier.apply(weights_init_classifier)
