import torch
import torch.nn as nn


def gen_ckpt_name(args):
    ckpt_name = "{}_{}_{}_{}_{}_{}_bn-{}_dropout-{}-{}_{}.pth.tar".format(
        args['dataset'], args['hdim'],
        args['nlayer'], args['policy'], args['noise_layer'],
        args['act_fn'], args['batchnorm'], args['input_drop'],
        args['hidden_drop'], args['rand_seed'])
    return ckpt_name

class IntermediateNoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100),
            activation_fn="relu", batchnorm='none', input_drop=0.,
            hidden_drop=0.):
        super(IntermediateNoisyMLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.n_cls = n_cls
        #batchnorm # ['none', 'before', 'after' (activation)]

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
        noise_layer = str(noise_layer)
        first_noise_layer = int(noise_layer[0])
        last_noise_layer = int(noise_layer[1:])
        print(" *** act_fn={} , first_noise_layer={}, last_noise_layer={}".format(self.activation_fn,
                                                                                  first_noise_layer,
                                                                                  last_noise_layer))
        layers = []
        if input_drop > 0:
            layers.append(nn.Dropout(input_drop))
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if first_noise_layer <= i < last_noise_layer:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                print("Layer{} is not updated".format(i))
            if batchnorm == 'none':
                layers += [layer, self.activation_fn]
            elif batchnorm == 'before':
                layers += [layer, nn.BatchNorm1d(dimh), self.activation_fn]
            else:
                layers += [layer, self.activation_fn, nn.BatchNorm1d(dimh)]
            if hidden_drop > 0:
                layers.append(nn.Dropout(hidden_drop))
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)

    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, type(self.activation_fn)):
                features.append(x)
        return features


class AlternatingNoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100),
            activation_fn="relu",
            batchnorm='none', input_drop=0., hidden_drop=0.):
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
        if input_drop > 0:
            layers.append(nn.Dropout(input_drop))
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if i % int(noise_layer) != 0:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                print("Layer{} is not updated".format(i))
            if batchnorm == 'none':
                layers += [layer, self.activation_fn]
            elif batchnorm == 'before':
                layers += [layer, nn.BatchNorm1d(dimh), self.activation_fn]
            elif batchnorm == 'after':
                layers += [layer, self.activation_fn, nn.BatchNorm1d(dimh)]
            else:
                raise ValueError('batchnorm should be one of {none,before,after} not {}'.format(batchnorm))
            if hidden_drop > 0:
                layers.append(nn.Dropout(hidden_drop))
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)

    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, type(self.activation_fn)):
                features.append(x)
        return features


class IncomingNoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100),
            activation_fn="relu", batchnorm='none', input_drop=0.,
            hidden_drop=0.):
        super(IncomingNoisyMLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.batchnorm = batchnorm
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
        if input_drop > 0:
            layers.append(nn.Dropout(input_drop))
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if i < int(noise_layer):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                print("Layer{} is not updated".format(i))
            if batchnorm == 'none':
                layers += [layer, self.activation_fn]
            elif batchnorm == 'before':
                layers += [layer, nn.BatchNorm1d(dimh), self.activation_fn]
            elif batchnorm == 'after':
                layers += [layer, self.activation_fn, nn.BatchNorm1d(dimh)]
            else:
                raise ValueError('batchnorm should be one of {none,before,after} not {}'.format(batchnorm))
            if hidden_drop > 0:
                layers.append(nn.Dropout(hidden_drop))
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)

    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, type(self.activation_fn)):
                features.append(x)
        return features


class OutgoingNoisyMLP(nn.Module):
    def __init__(self, input_dim, n_cls, noise_layer, layer_dims=(300,100),
            activation_fn="relu", batchnorm='none', input_drop=0., hidden_drop=0.):
        super(OutgoingNoisyMLP, self).__init__()
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
        if input_drop > 0:
            layers.append(nn.Dropout(input_drop))
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = input_dim if i==0 else layer_dims[i-1]
            layer = nn.Linear(inp_dim, dimh)
            if i >= int(noise_layer):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                print("Layer{} is not updated".format(i))
            if batchnorm == 'none':
                layers += [layer, self.activation_fn]
            elif batchnorm == 'before':
                layers += [layer, nn.BatchNorm1d(dimh), self.activation_fn]
            elif batchnorm == 'after':
                layers += [layer, self.activation_fn, nn.BatchNorm1d(dimh)]
            else:
                raise ValueError('batchnorm should be one of {none,before,after} not {}'.format(batchnorm))
            if hidden_drop > 0:
                layers.append(nn.Dropout(hidden_drop))
        layers.append(nn.Linear(self.layer_dims[-1], n_cls))

        self.output = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.output(x)

    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, type(self.activation_fn)):
                features.append(x)
        return features
