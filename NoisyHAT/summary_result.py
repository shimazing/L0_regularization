import numpy as np
import os
import torch
import argparse

from dataloaders import cifar as dataloader

os.system('scp -r server11@server11.mli.kr:/home/server11/mazing/L0_regularization/NoisyHAT/ckpt/*cifar* ckpt/')
os.system('scp -r server11@server11.mli.kr:/home/server11/mazing/L0_regularization/NoisyHAT/res/ res/')
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default='cifar', type=str)
parser.add_argument("--approach", default='noisy-hat', choices=['hat',
    'noisy-hat'])
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument("--nhid", default=100, type=int,required=False, help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--parameter", default="4.0,400", type=str)
args = parser.parse_args()

if args.output=='':
    if not os.path.exists("res"):
        os.makedirs("res")
    hyperparams = "_".join(args.parameter.split(","))
    args.output='./res/'+args.experiment+'_'+args.approach+"_"+str(args.seed)+ "_"+ hyperparams +'.txt'

acc = np.loadtxt(args.output)
print(acc)
#input()
'''
data,taskcla,inputsize=dataloader.get(seed=args.seed)

if args.approach=="noisy-hat":
    from approaches import noisy_hat as approach
    from networks import noisyalexnet_hat as network
else:
    from approaches import hat as approach
    from networks import alexnet_hat as network

print('Load data...')
data,taskcla,inputsize=dataloader.get(seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(inputsize,taskcla).cuda()
utils.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

hyperparams = "_".join(args.parameter.split(","))
ckpt_dir = 'ckpt'
for t in range(10):
    ckpt_namp = args.experiment + '_' + str(t) + "_" + args.approach + '_' +\
      str(args.nhid) + "_" + str(args.seed) + "_" + hyperparams + '.pth.tar'
    ckpt_filename = os.path.join(ckpt_dir, ckpt_name)

    ckpt = torch.load(ckpt_filename)
    test_acc = ckpt["test_acc"]
'''
