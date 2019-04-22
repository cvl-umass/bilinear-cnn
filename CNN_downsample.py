from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.model = models.resnet101(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 2048
        delattr(self.model, 'fc')

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

        return x

    def get_output_dim(self):
        return self.output_dim 

class ResNet_downsample(ResNet):
    def __init__(self, pool_fn='avg'):
        super(ResNet_downsample, self).__init__()
        if pool_fn == 'avg':
            self.downsample = nn.AvgPool2d(kernel_size=3, stride=2,
                                            padding=1, count_include_pad=False)
        elif pool_fn == 'max':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            exit()
        
class ResNet_layer0(ResNet_downsample):
    def __init__(self, pool_fn):
        super(ResNet_layer0, self).__init__(pool_fn)
        
    def forward(self, x):
        x = self.downsample(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        return x

class ResNet_layer1(ResNet_downsample):
    def __init__(self, pool_fn):
        super(ResNet_layer1, self).__init__(pool_fn)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.downsample(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        return x

class ResNet_layer2(ResNet_downsample):
    def __init__(self, pool_fn):
        super(ResNet_layer2, self).__init__(pool_fn)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.downsample(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        return x

class ResNet_layer3(ResNet_downsample):
    def __init__(self, pool_fn):
        super(ResNet_layer3, self).__init__(pool_fn)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.downsample(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        return x

class ResNet_layer4(ResNet_downsample):
    def __init__(self, pool_fn):
        super(ResNet_layer4, self).__init__(pool_fn)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.downsample(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        return x

class ResNet_layer5(ResNet_downsample):
    def __init__(self, pool_fn):
        super(ResNet_layer5, self).__init__(pool_fn)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.downsample(x)
        x = self.model.avgpool(x)

        return x

class CNN_Model(nn.Module):
    def __init__(self, num_classes, feature_dim, feature_extractors=None):
        super(CNN_Model, self).__init__()
        self.feature_extractors = feature_extractors
        self.feature_dim = feature_dim
        self.fc = nn.Linear(self.feature_dim, num_classes, bias=True)

    def forward(self, x): 
        x = self.feature_extractors(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        
        return y 

def create_cnn_model(model_name, num_classes, pool_fn='avg'):
    if model_name == 'resnet':
        feature_extractors = ResNet()
    elif  model_name == 'resnet_dsmp0':
        feature_extractors = ResNet_layer0(pool_fn)
    elif model_name == 'resnet_dsmp1':
        feature_extractors = ResNet_layer1(pool_fn)
    elif model_name == 'resnet_dsmp2':
        feature_extractors = ResNet_layer2(pool_fn)
    elif model_name == 'resnet_dsmp3':
        feature_extractors = ResNet_layer3(pool_fn)
    elif model_name == 'resnet_dsmp4':
        feature_extractors = ResNet_layer4(pool_fn)
    elif model_name == 'resnet_dsmp5':
        feature_extractors = ResNet_layer5(pool_fn)
    else:
        exit()
    '''
    if input_size != 224:
        assert model_name == 'resnet' 
    if model_name == 'vgg':
        feature_extractors = VGG()
    elif model_name == 'resnet':
        feature_extractors = ResNet(input_size)
    elif model_name == 'densenet':
        feature_extractors = DenseNet()
    else:
        exit()
    '''

    output_dim = feature_extractors.get_output_dim()
    return CNN_Model(num_classes, output_dim, feature_extractors)
