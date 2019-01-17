import numpy as np
import os
import torch
import argparse
from dataloaders import cifar as dataloader
import utils
import matplotlib.pyplot as plt
os.system('scp -r server11@server11.mli.kr:/home/server11/mazing/L0_regularization/NoisyHAT/res/ res/')
os.system('scp -r server11@server11.mli.kr:/home/server11/mazing/L0_regularization/NoisyHAT/ckpt/*cifar* ckpt/')
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default='cifar', type=str)
#parser.add_argument("--approach", default='noisy-hat', choices=['hat',
#    'noisy-hat'])
#parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument("--nhid", default=100, type=int,required=False, help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--parameter", default="4.0,400", type=str)
args = parser.parse_args()


result = {'noisy-hat': dict({}), 'hat': dict({})}
for approach_ in ['noisy-hat', 'hat']:
 if approach_ == "noisy-hat":
    from approaches import noisy_hat as approach
    from networks import noisyalexnet_hat as network
 else:
    from approaches import hat as approach
    from networks import alexnet_hat as network
 data,taskcla,inputsize=dataloader.get(seed=0)
 net=network.Net(inputsize,taskcla).cuda()
 appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args)
 for lambda_ in  [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]:
  for smax in [25, 50, 100, 200, 400, 800]:
    parameter = "{},{}".format(lambda_, smax)
    result[approach_][parameter] = dict({})
    capacity_list = []
    acc_list = []
    for seed in range(4):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        hyperparams = "_".join(parameter.split(","))
        output='./res/'+args.experiment+'_'+approach_+"_"+str(seed)+ "_"+ hyperparams +'.txt'

        capacity = np.zeros((len(taskcla), len(taskcla)))
        try:
            acc = np.loadtxt(output)
            print(output, '# Load')
        except:
            print(output, '# No acc')
            continue

        ckpt_dir = 'ckpt'
        for t in range(10):
            ckpt_name = args.experiment + '_' + str(t) + "_" + approach_ + '_' +\
              str(args.nhid) + "_" + str(seed) + "_" + hyperparams + '.pth.tar'
            ckpt_filename = os.path.join(ckpt_dir, ckpt_name)

            thrs = 1e-4
            try:
                ckpt = torch.load(ckpt_filename)
                print(ckpt_filename, '# Load')
            except:
                print(ckpt_filename, 'no ckpt')
                continue

            try:
                net.load_state_dict(ckpt["model_state_dict"])
            except:
                print("network loading fail")
                continue

            def apply_thrs(args):
                return [(arg > thrs).float() for arg in args]

            total = 0
            total += torch.ones_like(net.c1.weight).data.sum().item()
            total += torch.ones_like(net.c1.bias).data.sum().item()
            total += torch.ones_like(net.c2.weight).data.sum().item()
            total += torch.ones_like(net.c2.bias).data.sum().item()
            total += torch.ones_like(net.c3.weight).data.sum().item()
            total += torch.ones_like(net.c3.bias).data.sum().item()
            total += torch.ones_like(net.fc1.weight).data.sum().item()
            total += torch.ones_like(net.fc1.bias).data.sum().item()
            total += torch.ones_like(net.fc2.weight).data.sum().item()
            total += torch.ones_like(net.fc2.bias).data.sum().item()
            total += torch.ones_like(net.last[t].weight).data.sum().item()
            total += torch.ones_like(net.last[t].bias).data.sum().item()
            for u in range(0, t+1):
                #gc1, gc2, gc3, gfc1, gfc2 = net.mask(t, s=400)
                gc1, gc2, gc3, gfc1, gfc2 = apply_thrs(net.mask(torch.LongTensor([u]).cuda(), s=400))
                # c1
                nonzero = gc1.data.view(-1,1,1,1).expand_as(net.c1.weight).sum().item()
                nonzero += gc1.data.view(-1).sum().item()
                # c2
                post=gc2.data.view(-1,1,1,1).expand_as(net.c2.weight)
                pre=gc1.data.view(1,-1,1,1).expand_as(net.c2.weight)
                nonzero += torch.min(post,pre).sum().item()
                nonzero += gc2.data.view(-1).sum().item()
                # c3
                post=gc3.data.view(-1,1,1,1).expand_as(net.c3.weight)
                pre=gc2.data.view(1,-1,1,1).expand_as(net.c3.weight)
                nonzero += torch.min(post,pre).sum().item()
                nonzero += gc3.data.view(-1).sum().item()
                #fc1
                post=gfc1.data.view(-1,1).expand_as(net.fc1.weight)
                pre=gc3.data.view(-1,1,1).expand((net.ec3.weight.size(1),net.smid,net.smid)).contiguous().view(1,-1).expand_as(net.fc1.weight)
                nonzero += torch.min(post,pre).sum().item()
                nonzero += gfc1.data.view(-1).sum().item()
                # fc2
                post=gfc2.data.view(-1,1).expand_as(net.fc2.weight)
                pre=gfc1.data.view(1,-1).expand_as(net.fc2.weight)
                nonzero += torch.min(post,pre).sum().item()
                nonzero += gfc2.data.view(-1).sum().item()
                # last
                nonzero += gfc2.data.view(1, -1).expand_as(net.last[t].weight).sum().item()
                nonzero += torch.ones_like(net.last[t].bias).data.sum().item()
                capacity[t, u] = nonzero / total
        #print("=========== Capacity ===========")
        #print(capacity[-1])
        capacity_list.append(capacity)
        acc_list.append(acc)
    if capacity_list:
        result[approach_][parameter]['capacity'] = np.array(capacity_list).mean(axis=0)
    else:
        result[approach_][parameter]['capacity'] = np.zeros((10, 10))
    result[approach_][parameter]['capacity_list'] = capacity_list
    if acc_list:
        result[approach_][parameter]['acc'] = np.array(acc_list).mean(axis=0)
    else:
        result[approach_][parameter]['acc'] = np.zeros((10, 10))
    result[approach_][parameter]['acc_list'] = acc_list

np.set_printoptions(precision=4)

capacity2accuracy = {'noisy-hat': [[] for _ in range(10)],
                     'hat': [[] for _ in range(10)]}

for approach_ in ['noisy-hat', 'hat']:
    for param in result[approach_]:
        for task in range(10):
            capacity2accuracy[approach_][task].append(
                (result[approach_][param]['capacity'][-1][task],
                 result[approach_][param]['acc'][-1][task])
            )

fig, ax = plt.subplots(10, 1)
for task in range(10):
    ax[task].set_ylabel('ACC')
    ax[task].set_xlabel('Capacity')
    ax[task].set_title('task {}'.format(task))
    capacity, acc = zip(*capacity2accuracy['noisy-hat'][task])
    ax[task].scatter(capacity, acc, color="r", label='noisy-hat')
    capacity, acc = zip(*capacity2accuracy['hat'][task])
    ax[task].scatter(capacity, acc, color="b", label='hat')
    ax[task].legend()
plt.show()
