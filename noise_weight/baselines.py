import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

from models import NoisyMLP
from torchvision import datasets, transforms
import copy
import argparse
import os
import numpy as np
from dataloader import *
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--model', default='logisticRegression', type=str,
                    help='name of experiment')
parser.add_argument("--save_dir", type=str, default="ckpt")
parser.add_argument("--policy", type=str, default="NoisyMLP")
parser.add_argument("--rand_seed", type=int, default=11)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", choices=['wine', 'mnist', 'abalone', 'redwine',
   'letter','dna','protein','satimage','shuttle', 'ijcnn1'], default='wine')
parser.add_argument("--C", default=1., type=float)
parser.add_argument("--kernel", default='rbf', choices=['linear', 'rbf'])
parser.add_argument("--metric", default='auc', choices=['auc', 'acc'])

args = parser.parse_args()
CKPT_DIR = "ckpt"
DATA_DIR = "../data"
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}




def main():
    if args.model == 'logisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2', C=args.C, class_weight='balanced',
                random_state=args.rand_seed)
        '''
        LogisticRegression(penalty=’l2’, dual=False,
        tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
        random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0,
        warm_start=False, n_jobs=None)'''
    elif args.model == 'svm':
        from sklearn.svm import SVC
        model = SVC(C=args.C, kernel=args.kernel, random_state=args.rand_seed,
                probability=True, gamma='scale')
        '''
        SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’,
        coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
        class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’,
        random_state=None)
        '''
    ckpt_name = "{}_{}_{}_{}_{}.pth.tar".format(args.dataset, args.model, args.C,
            args.kernel, args.rand_seed)
    np.random.seed(args.rand_seed)
    #torch.manual_seed(args.rand_seed)
    #if args.cuda:
    #    torch.cuda.manual_seed_all(args.rand_seed)
    # create model
    #token = args.model.split("-")
    #layer_dims = tuple([int(elem) for elem in token[1:]])
    #print(len(layer_dims), layer_dims, args.policy, args.C, args.act_fn)
    # data
    if args.dataset in ['wine', 'abalone', 'mnist', 'redwine']:
        if args.dataset == 'wine':
            train_set = WhiteWine(args.rand_seed, train=True, test_ratio=0.3)
            test_set = WhiteWine(args.rand_seed, train=False, test_ratio=0.3)
            input_dim = train_set.X.shape[1]
            n_cls = 7
        elif args.dataset == 'redwine':
            train_set = RedWine(seed=args.rand_seed, train=True, test_ratio=0.3)
            test_set = RedWine(seed=args.rand_seed, train=False, test_ratio=0.3)
            input_dim = train_set.X.shape[1]
            n_cls = train_set.n_cls
        elif args.dataset == 'abalone':
            train_set = Abalone(seed=args.rand_seed, train=True, test_ratio=0.3)
            test_set = Abalone(seed=args.rand_seed, train=False, test_ratio=0.3)
            input_dim = train_set.X.shape[1]
            n_cls = train_set.n_cls
        elif args.dataset == 'mnist':
            normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            n_cls = 10
            dset_string = "datasets.MNIST"
            train_tfms = []
            train_tfms += [transforms.ToTensor(), normalize]
            train_set = eval(dset_string)(root=DATA_DIR, train=True,
                                          transform=transforms.Compose(train_tfms), download=True)

            valid_set = eval(dset_string)(root=DATA_DIR, train=True,
                                          transform=transforms.Compose(train_tfms), download=True)

            test_set = eval(dset_string)(root=DATA_DIR, train=False,
                                         transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)
        else:
            print("No dataset called '{}'".format(args.dataset))
            raise Exception

        train_X, train_y = train_set.X.numpy(), train_set.y.numpy()
        test_X, test_y = test_set.X.numpy(), test_set.y.numpy()
        num_train = len(train_set)
        indices = list(range(num_train))
        valid_size = 0.25
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_X, train_y, valid_X, valid_y = train_X[train_idx], train_y[train_idx], train_X[valid_idx], train_y[valid_idx]
    elif args.dataset in ['letter','dna','protein','satimage','shuttle',
            'ijcnn1']:
        train_set = eval(args.dataset.upper())(train=True, val=False)
        valid_set = eval(args.dataset.upper())(train=True, val=True)
        test_set = eval(args.dataset.upper())(train=False)
        train_X, train_y = train_set.X.numpy(), train_set.y.numpy()
        valid_X, valid_y = valid_set.X.numpy(), valid_set.y.numpy()
        test_X, test_y = test_set.X.numpy(), test_set.y.numpy()
        n_cls = train_set.n_cls
        n_features = train_set.n_features

    model.fit(train_X, train_y) # learning
    train_pred = model.predict(train_X)
    train_prob = model.predict_proba(train_X)
    valid_pred = model.predict(valid_X)
    valid_prob = model.predict_proba(valid_X)
    test_pred = model.predict(test_X)
    test_prob = model.predict_proba(test_X)

    train_acc = (train_pred == train_y).astype(np.float32).mean()
    valid_acc = (valid_pred == valid_y).astype(np.float32).mean()
    test_acc = (test_pred == test_y).astype(np.float32).mean()

    train_y_ = np.eye(n_cls)[train_y.astype(int)]
    binary  = []
    for i in range(n_cls):
        if len(np.unique(train_y_[:, i])) == 2:
            binary.append(i)
    train_auc = roc_auc_score(train_y_[:, binary], train_prob[:, binary], 'macro')

    valid_y_ = np.eye(n_cls)[valid_y.astype(int)]
    binary  = []
    for i in range(n_cls):
        if len(np.unique(valid_y_[:, i])) == 2:
            binary.append(i)
    valid_auc = roc_auc_score(valid_y_[:, binary], valid_prob[:, binary], 'macro')

    test_y_ = np.eye(n_cls)[test_y.astype(int)]
    binary  = []
    for i in range(n_cls):
        if len(np.unique(test_y_[:, i])) == 2:
            binary.append(i)
    test_auc = roc_auc_score(test_y_[:, binary], test_prob[:, binary], 'macro')

    state = {
                "model": args.model,
                "kernel": args.kernel,
                "C": args.C,
                'fitted_model': model,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'valid_auc': valid_auc,
                'valid_acc': valid_acc,
                'test_auc': test_auc,
                'test_acc': test_acc,
            }
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, ckpt_name)
    torch.save(state, filename)
    print('[{}] Valid Accuracy: {:.2f}'.format(ckpt_name, valid_acc * 100))
    print('[{}] Valid AUC: {:.2f}'.format(ckpt_name, valid_auc))
    print('[{}] Test Accuracy: {:.2f}'.format(ckpt_name, test_acc * 100))
    print('[{}] Test AUC: {:.2f}'.format(ckpt_name, test_auc))

if __name__ == '__main__':
    main()
