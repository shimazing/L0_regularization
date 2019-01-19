import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, hdim=4096):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096), # for imagenet
            nn.Linear(512 * 1 * 1, hdim), # for cifar
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hdim, num_classes),
        )
        #self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers_with_noise(noise_cfg, batch_norm=False, in_channels=3):
    layers = []
    for v, requires_grad in noise_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d.weight.requires_grad = requires_grad
            conv2d.bias.requires_grad = requires_grad
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


noise_cfg = {
    0: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, True), (256, True), (256, True), ('M', None),
        (512, True), (512, True), (512, True), ('M', None),
        (512, True), (512, True), (512,True), ('M', None)],
    1: [(64, True), (64, True), ('M', None),
        (128, False), (128, False), ('M', None),#
        (256, True), (256, True), (256, True), ('M', None),
        (512, True), (512, True), (512, True), ('M', None),
        (512, True), (512, True), (512,True), ('M', None)],
    2: [(64, True), (64, True), ('M', None),
        (128, False), (128, False), ('M', None),#
        (256, False), (256, False), (256, False), ('M', None),#
        (512, True), (512, True), (512, True), ('M', None),
        (512, True), (512, True), (512, True), ('M', None)],
    3: [(64, True), (64, True), ('M', None),
        (128, False), (128, False), ('M', None),#
        (256, False), (256, False), (256, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None),#
        (512, True), (512, True), (512,True), ('M', None)],
    4: [(64, True), (64, True), ('M', None),
        (128, False), (128, False), ('M', None),#
        (256, False), (256, False), (256, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None)],#
    5: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, False), (256, False), (256, False), ('M', None),#
        (512, True), (512, True), (512, True), ('M', None),
        (512, True), (512, True), (512, True), ('M', None)],
    6: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, False), (256, False), (256, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None),#
        (512, True), (512, True), (512, True), ('M', None)],
    7: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, False), (256, False), (256, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None)],#
    8: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, True), (256, True), (256, True), ('M', True),
        (512, False), (512, False), (512, False), ('M', None),#
        (512, True), (512, True), (512, True), ('M', None)],
    9: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, True), (256, True), (256, True), ('M', True),
        (512, False), (512, False), (512, False), ('M', None),#
        (512, False), (512, False), (512, False), ('M', None)],#
    10: [(64, True), (64, True), ('M', None),
        (128, True), (128, True), ('M', None),
        (256, True), (256, True), (256, True), ('M', True),
        (512, True), (512, True), (512, False), ('M', True),
        (512, False), (512, False), (512, False), ('M', None)]#
}


def vgg11(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)

def vgg16_with_noise(key, bn=False, num_classes=100, hdim=4096, in_channels=3):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers_with_noise(noise_cfg[key], batch_norm=bn, in_channels=in_channels),
            num_classes=num_classes, hdim=hdim)
    def init_weight(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound,bound)
    model.apply(init_weight)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model

def vgg19(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

