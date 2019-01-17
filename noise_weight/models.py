import torch
import torch.nn as nn


class NoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100), activation_fn="relu"):
        super(NoisyMLP, self).__init__()
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
            layers += [layer, self.activation_fn]
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
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
