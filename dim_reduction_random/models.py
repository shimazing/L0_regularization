import torch
import torch.nn as nn
from gram_schmidt import gs


class NoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, layer_dims=(300,100), rank=None):
        super(NoisyMLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.n_cls = n_cls
        if rank is None: # full rank
            rank = layer_dims[-1]
        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if i == 1:
                weight = layer.weight.t().data.numpy() # in x out
                orthogonal = gs(X)
                orthogonal = orthogonal[:rank].T # out x rank
                coef = np.random.uniform(-1, 1, size=(rank, inp_dim))
                #np.matmul(orthogonal, coef) # out x in
                weight[:, :] = np.matmul(orthogonal, coef)
            layers += [layer, nn.ReLU()]
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
