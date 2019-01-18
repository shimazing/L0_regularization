import torch
import torch.nn as nn
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AlternatingNoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100), activation_fn="relu"):
        super(AlternatingNoisyMLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.n_cls = n_cls

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation_fn == "softplus":
            self.activation_fn = nn.Softplus()
        else:
            raise ValueError
        print(" *** act_fn={} , noise_layer={}".format(self.activation_fn, noise_layer))

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if i % 2 == noise_layer:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                print("Layer{} is not updated".format(i))
            layers += [layer, self.activation_fn]
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)

class AlternatingNoisyCNN(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer=0, n_conv=2, conv_dim=5, fc_layer_dims=[],
            activation_fn='relu'):
        super(AlternatingNoisyCNN, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation_fn == "softplus":
            self.activation_fn = nn.Softplus()
        else:
            raise ValueError
        print(" *** act_fn={} , noise_layer={}, n_conv={}, fc_layer_dims={}".format(
            self.activation_fn, noise_layer, n_conv, fc_layer_dims))

        layers = []
        in_channels = input_dim[0]
        wh_dim = input_dim[1]
        for i in range(n_conv):
            layers += [nn.Conv2d(in_channels, conv_dim*np.power(2, i), 5, padding=2),
                      self.activation_fn, nn.MaxPool2d(2)]
            if i % 2 == noise_layer:
                layers[-3].weight.requires_grad = False
                layers[-3].bias.requires_grad = False
                print("Conv Layer {} is not Uptated".format(i))
            wh_dim //= 2
            in_channels = conv_dim * np.power(2, i)
        layers.append(Flatten())
        layers.append(nn.Linear(wh_dim * wh_dim * in_channels, n_cls))
        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        return self.output(x)


class IncomingNoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100), activation_fn="relu"):
        super(IncomingNoisyMLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.n_cls = n_cls

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation_fn == "softplus":
            self.activation_fn = nn.Softplus()
        else:
            raise ValueError
        print(" *** act_fn={} , noise_layer={}".format(self.activation_fn, noise_layer))

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if i <= noise_layer:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                print("Layer{} is not updated".format(i))
            layers += [layer, self.activation_fn]
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)

class IncomingNoisyCNN(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, n_conv=2, conv_dim=5, fc_layer_dims=[],
            activation_fn='relu'):
        super(IncomingNoisyCNN, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation_fn == "softplus":
            self.activation_fn = nn.Softplus()
        else:
            raise ValueError
        print(" *** act_fn={} , noise_layer={}, n_conv={}, fc_layer_dims={}".format(
            self.activation_fn, noise_layer, n_conv, fc_layer_dims))

        layers = []
        in_channels = input_dim[0]
        wh_dim = input_dim[1]
        for i in range(n_conv):
            layers += [nn.Conv2d(in_channels, conv_dim*np.power(2, i), 5, padding=2),
                      self.activation_fn, nn.MaxPool2d(2)]
            if i <= noise_layer:
                layers[-3].weight.requires_grad = False
                layers[-3].bias.requires_grad = False
                print("Conv Layer {} is not Uptated".format(i))
            wh_dim //= 2 # maxpool
            in_channels = conv_dim * np.power(2, i)
        layers.append(Flatten())
        layers.append(nn.Linear(wh_dim * wh_dim * in_channels, n_cls))
        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        return self.output(x)



class NoisyMLP1(nn.Module):
    def __init__(self, input_dim, n_cls, layer_dims=(300,100)):
        super(NoisyMLP1, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.n_cls = n_cls

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if i == 0:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
            layers += [layer, nn.ReLU()]
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)
