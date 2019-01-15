import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import utils

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_noise = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.bias = Parameter(torch.Tensor(out_features))
        self.bias_noise = Parameter(torch.Tensor(out_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        # param initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        # noise param initialization
        init.kaiming_uniform_(self.weight_noise, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_noise)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias_noise, -bound, bound)

    def forward(self,x, x_noise=None):
        if x_noise is None:
            x_noise = x
        output_origin = F.linear(x, self.weight, self.bias)
        output_noise = F.linear(x_noise, self.weight_noise, self.bias_noise)
        return output_origin, output_noise

class Net(torch.nn.Module):
    def __init__(self,inputsize,taskcla,nlayers=2,nhid=2000,pdrop1=0.2,pdrop2=0.5):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.nlayers=nlayers

        self.relu=torch.nn.ReLU()
        #self.drop1=torch.nn.Dropout(pdrop1)
        #self.drop2=torch.nn.Dropout(pdrop2)
        #self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc1 = NoisyLinear(ncha*size*size,nhid)
        self.efc1=torch.nn.Embedding(len(self.taskcla),nhid)
        if nlayers>1:
            #self.fc2=torch.nn.Linear(nhid,nhid)
            self.fc2=NoisyLinear(nhid,nhid)
            self.efc2=torch.nn.Embedding(len(self.taskcla),nhid)
        if nlayers>2:
            #self.fc3=torch.nn.Linear(nhid,nhid)
            self.fc3=NoisyLinear(nhid,nhid)
            self.efc3=torch.nn.Embedding(len(self.taskcla),nhid)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            #self.last.append(torch.nn.Linear(nhid,n))
            self.last.append(NoisyLinear(nhid,n))

        self.gate=torch.nn.Sigmoid()
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        else:
            raise ValueError
        # Gated
        #h=self.drop1(x.view(x.size(0),-1))  # Original code, applying dropout on input data and hidden layers
        #h=self.drop2(self.relu(self.fc1(h)))
        h = x.view(x.size(0),-1)
        output1_origin, output1_noise = self.fc1(h)
        output1 = output1_origin.mul_(gfc1.abs().sign()) + output1_noise
        h1 = self.relu(output1)
        h_origin, h_noise = h1.mul_(gfc1), h1
        if self.nlayers>1:
            output2_origin, output2_noise = self.fc2(h_origin, h_noise)
            output2 = output2_origin.mul_(gfc2.abs().sign()) + output2_noise
            h2 = self.relu(output2)
            h_origin, h_noise = h2.mul_(gfc2), h2
            #h=self.drop2(self.relu(self.fc2(h)))
            #h=h*gfc2.expand_as(h)
        if self.nlayers>2:
            output3_origin, output3_noise = self.fc3(h_origin, h_noise)
            output3 = output3_origin.mul_(gfc3.abs().sign()) + output3_noise
            h3 = self.relu(output3)
            h_origin, h_noise = h3.mul_(gfc3), h3
            #h=self.drop2(self.relu(self.fc3(h)))
            #h=h*gfc3.expand_as(h)
        y=[]
        for t,i in self.taskcla:
            output_origin, output_noise = self.last[t](h_origin, h_noise)
            y.append(output_origin + output_noise)
        return y, masks

    def mask(self,t,s=1):
        gfc1=self.gate(s*self.efc1(t))
        if self.nlayers==1: return gfc1
        gfc2=self.gate(s*self.efc2(t))
        if self.nlayers==2: return [gfc1,gfc2]
        gfc3=self.gate(s*self.efc3(t))
        return [gfc1,gfc2,gfc3]

    def get_view_for(self,name,masks):
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        if name == 'fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif name == 'fc1.bias':
            return gfc1.data.view(-1)
        elif name == 'fc2.weight':
            post = gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre = gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif name == 'fc2.bias':
            return gfc2.data.view(-1)
        elif name == 'fc3.weight':
            post = gfc3.data.view(-1,1).expand_as(self.fc3.weight)
            pre = gfc2.data.view(1,-1).expand_as(self.fc3.weight)
            return torch.min(post,pre)
        elif name == 'fc3.bias':
            return gfc3.data.view(-1)
        else:
            return None

