from models.block import *


# Define the ResNet50-based Model
class ft_resnet50(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the DenseNet121-based Model
class ft_densenet121(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_resnet50mid(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x0 = self.model.avgpool(x)  # x0  n*1024*1*1
        x = self.model.layer4(x)
        x1 = self.model.avgpool(x)  # x1  n*2048*1*1
        x = torch.cat((x0, x1), 1)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the ResNet101-based Model
class ft_resnet101(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.resnet101(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the ResNet152-based Model
class ft_resnet152(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.resnet152(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the DenseNet169-based Model
class ft_densenet169(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(1664, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the DenseNet201-based Model
class ft_densenet201(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet201(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(1920, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Define the DenseNet161-based Model
class ft_densenet161(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet161(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(2208, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = self.classifier(x)
        return x, y


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()
        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        # define 6 classifiers
        for i in range(self.part):
            setattr(self, 'classifier' + str(i), ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)

        features = x.view(x.size(0), x.size(1), x.size(2))
        x = self.dropout(x)

        # get six part feature batchsize*2048*6
        outputs = []
        for i in range(self.part):
            features.append(torch.squeeze(x[:, :, i]))
            c = getattr(self, 'classifier' + str(i))
            outputs.append(c(features[i]))

        sm = nn.Softmax(dim=1)
        outputs = sum([sm(outputs[i]) for i in range(self.part)])

        return features, outputs
