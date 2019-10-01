from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.model = models.resnet101(pretrained=True)
        self.input_size = 224
        delattr(self.model, 'fc')
        delattr(self.model, 'avgpool')

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.input_size = 224

    def forward(self, x):
        x = self.model.features(x)
        return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # self.model = models.vgg16(pretrained=True)
        self.model = models.vgg16(pretrained=True).features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.input_size = 224

    def forward(self, x):
        x = self.model(x)
        # x = self.model.features(x)
        return x

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.model = models.densenet201(pretrained=True)
        self.input = 224

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        return x

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.input_size = 299

    def forward(self, x):
        if self.model.transform_input:
            '''
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            '''
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)

        return x
