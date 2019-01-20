import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import time
from models_yki import AlternatingNoisyCNN, IncomingNoisyCNN
from torchvision import datasets, transforms
import copy
import argparse
import os
import numpy as np
from dataloader import *
from vggs import vgg16_with_noise
from wide_resnet import WideResNet

parser = argparse.ArgumentParser(description='NoisyNet MLP-300-300 Training')
parser.add_argument('--max_epoch', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument("--act_fn", type=str, default="relu")
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
#parser.add_argument('--model', default='MLP-300-300-300-300', type=str,
#                    help='name of experiment')
parser.add_argument("--save_dir", type=str, default="ckpt")
parser.add_argument("--policy", type=str, required=True,
        choices=["AlternatingNoisyCNN", "IncomingNoisyCNN", "NoisyVgg16",
        "NoisyWideResNet"])
parser.add_argument("--noise_layer", type=int, required=True, help="-1 means training whole networks")
parser.add_argument("--n_conv", type=int, default=2)
parser.add_argument("--conv_dim", type=int, default=10)
parser.add_argument("--hdim", type=int, default=4096, choices=[4096, 2048, 1024, 512])
parser.add_argument("--rand_seed", type=int, default=11)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--augment", action="store_true", default=False)
parser.add_argument("--batchnorm", action="store_true", default=False)
parser.add_argument("--dataset", type=str, required=True,
                    choices=["cifar10", "cifar100", "fashionmnist", "svhn"])# "mnist",
                             #"whitewine", "redwine", "abalone"])
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
CKPT_DIR = "ckpt"
DATA_DIR = "../data"
EARLY_STOPPING_CRITERION = args.max_epoch  # If there is no learning improvement withing the consecutive N epochs, stopping
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
SAVE_EPOCH = 100


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)

def main():
    if args.policy == "NoisyVgg16":
        args.n_conv = 'na'
        args.conv_dim = 'na'
        if args.batchnorm:
            args.policy += "BN"
        args.policy += "-{}".format(args.hdim)
    elif args.policy == "NoisyWideResNet":
        args.n_conv = 'na'
        args.conv_dim = 'na'

    ckpt_name = "{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, args.policy,
                                                   args.n_conv, args.conv_dim,
                                                   args.noise_layer, args.act_fn, args.rand_seed)
    if args.augment:
        ckpt_name = 'augment_' + ckpt_name
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.rand_seed)

    #-------------------------------------------------------------------------------------------------------------------
    # Load dataset
    if args.dataset == "letter":
        train_set = Letter(train=True, val=False)
        valid_set = Letter(train=True, val=True)
        test_set = Letter(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif args.dataset == "dna":
        train_set = DNA(train=True, val=False)
        valid_set = DNA(train=True, val=True)
        test_set = DNA(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif args.dataset == "satimage":
        train_set = Satimage(train=True, val=False)
        valid_set = Satimage(train=True, val=True)
        test_set = Satimage(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif args.dataset == "shuttle":
        train_set = Shuttle(train=True, val=False)
        valid_set = Shuttle(train=True, val=True)
        test_set = Shuttle(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif args.dataset == "ijcnn1":
        train_set = IJCNN1(train=True, val=False)
        valid_set = IJCNN1(train=True, val=True)
        test_set = IJCNN1(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif args.dataset == "mnist":
        normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        n_features = 784
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
    elif args.dataset.startswith("cifar"):
        n_features = 3*32*32
        n_cls = 10 if args.dataset =="cifar10" else 100
        dset_string = 'datasets.CIFAR10' if args.dataset == 'cifar10' else 'datasets.CIFAR100'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_tfms = [transforms.ToTensor(), normalize]
        valid_set = eval(dset_string)(root='../data', train=True, transform=transforms.Compose(train_tfms),
                                      download=True)
        test_set = eval(dset_string)(root='../data', train=False, transform=transforms.Compose(train_tfms),
                                     download=True)
        if args.augment:
            train_tfms = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip()] + train_tfms
        train_set = eval(dset_string)(root='../data', train=True, transform=transforms.Compose(train_tfms),
                                      download=True)
    elif args.dataset == "fashionmnist":
        dset_string = "datasets.FashionMNIST"
        n_features = 784
        n_cls = 10
        train_tfms = []
        train_tfms += [transforms.Pad(2), transforms.ToTensor()]#, normalize]
        Fashion_DATA_DIR = os.path.join(DATA_DIR, "FashionMNIST")
        train_set = eval(dset_string)(root=Fashion_DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)

        valid_set = eval(dset_string)(root=Fashion_DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)

        test_set = eval(dset_string)(root=Fashion_DATA_DIR, train=False,
                                     transform=transforms.Compose([transforms.ToTensor()]), download=True)
    elif args.dataset == "svhn":
        dset_string = "datasets.SVHN"
        n_features = 3*32*32
        svhn_DATA_DIR = os.path.join(DATA_DIR, "SVHN")
        n_cls = 10
        train_tfms = [transforms.ToTensor(),
                      # transforms.Lambda(lambda x: x.mul_(1./255))]
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        train_set = eval(dset_string)(root=svhn_DATA_DIR, split='train',
                                      transform=transforms.Compose(train_tfms),
                                      download=True)
        valid_set = eval(dset_string)(root=svhn_DATA_DIR, split='train',
                                      transform=transforms.Compose(train_tfms), download=True)
        test_set = eval(dset_string)(root=svhn_DATA_DIR, split='test',
                                     transform=transforms.Compose(train_tfms), download=True)
    elif args.dataset == "redwine":
        train_set = RedWine(train=True)
        valid_set = RedWine(train=True)
        test_set = RedWine(train=False)
        n_cls = test_set.n_cls
        n_features = test_set.n_features
    elif args.dataset == "whitewine":
        train_set = WhiteWine(train=True)
        valid_set = WhiteWine(train=True)
        test_set = WhiteWine(train=False)
        n_cls = test_set.n_cls
        n_features = test_set.n_features
    elif args.dataset == "abalone":
        train_set = Abalone(train=True)
        valid_set = Abalone(train=True)
        test_set = Abalone(train=False)
        n_cls = test_set.n_cls
        n_features = test_set.n_features
    else:
        raise ValueError

    #------------------------------------------------------------------------------------------------------------------
    # data loader
    if args.dataset in ["mnist", "cifar10", "cifar100", "whitewine", "redwine",
            "abalone", "redwine", "fashionmnist", "svhn"]:
        # For those data where validation set is not given
        num_train = len(train_set)
        indices = list(range(num_train))
        valid_size = 0.25
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   sampler=train_sampler, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
                                                   sampler=valid_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset in ["letter", "dna", "satimage", "shuttle", "ijcnn1"]:
        # For those data where validation set is given
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
                                                   shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, **kwargs)
    else:
        raise ValueError

    # create model
    #token = args.model.split("-")
    #layer_dims = tuple([int(elem) for elem in token[1:]])
    #print(args.dataset, len(layer_dims), layer_dims, args.policy, args.noise_layer)

    if args.policy == "AlternatingNoisyCNN":
        model = AlternatingNoisyCNN(input_dim=(3, 32, 32), n_cls=n_cls,
                n_conv=args.n_conv, conv_dim=args.conv_dim, fc_layer_dims=[],
                    activation_fn=args.act_fn, noise_layer=args.noise_layer)
    elif args.policy == "IncomingNoisyCNN":
        model = IncomingNoisyCNN(input_dim=(3, 32, 32), n_cls=n_cls,
                n_conv=args.n_conv, conv_dim=args.conv_dim, fc_layer_dims=[],
                activation_fn=args.act_fn, noise_layer=args.noise_layer)
    elif args.policy.startswith("NoisyVgg16"):
        model = vgg16_with_noise(args.noise_layer, bn=args.batchnorm, num_classes=n_cls,
                hdim=args.hdim, in_channels=1 if args.dataset=='fashionmnist' else 3)
    elif args.policy == "NoisyWideResNet":
        model = WideResNet(28, 10, 0.3, n_cls, noise_layer=args.noise_layer,
                in_channels=1 if args.dataset=='fashionmnist' else 3)
    else:
        raise ValueError
    #
    param_list, name_list, noise_param_list, noise_name_list = [], [], [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            name_list.append(name)
        else:
            noise_param_list.append(param)
            noise_name_list.append(name)
    #
    if args.policy == "NoisyWideResNet":
        optimizer = torch.optim.SGD(param_list, lr=learning_rate(0.1, 0),
                momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(param_list, args.lr)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # optionally resume from a checkpoint
    if os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
        checkpoint = torch.load(os.path.join(CKPT_DIR, ckpt_name))
        epoch = start_epoch = checkpoint['epoch']
        test_acc = checkpoint['test_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        train_loss_list = checkpoint["train_loss_list"]
        train_acc_list = checkpoint["train_acc_list"]
        train_auc_list = checkpoint["train_auc_list"]
        valid_loss_list = checkpoint["valid_loss_list"]
        valid_acc_list = checkpoint["valid_acc_list"]
        valid_auc_list = checkpoint["valid_auc_list"]
        non_zero_list = checkpoint["non_zero_list"]
        forward_time = checkpoint["forward_time"]
        backward_time = checkpoint["backward_time"]
        optim_time = checkpoint["optim_time"]
        non_zero = non_zero_list[-1]
        best_valid_acc = max(valid_acc_list)
        best_valid_auc = max(valid_auc_list)
        print(" *** Resume: [{}] Test Acc: {:.2f} Auc: {:.4f}"
              " at epoch: {} ***".format(ckpt_name, checkpoint["test_acc"]*100,
                                        checkpoint["test_auc"], checkpoint["epoch"]))
    elif os.path.exists(os.path.join(CKPT_DIR, "best_" + ckpt_name)):
        checkpoint = torch.load(os.path.join(CKPT_DIR, "best_" + ckpt_name))
        epoch = start_epoch = checkpoint['epoch']
        test_acc = checkpoint['test_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        train_loss_list = checkpoint["train_loss_list"]
        train_acc_list = checkpoint["train_acc_list"]
        train_auc_list = checkpoint["train_auc_list"]
        valid_loss_list = checkpoint["valid_loss_list"]
        valid_acc_list = checkpoint["valid_acc_list"]
        valid_auc_list = checkpoint["valid_auc_list"]
        non_zero_list = checkpoint["non_zero_list"]
        forward_time = checkpoint["forward_time"]
        backward_time = checkpoint["backward_time"]
        optim_time = checkpoint["optim_time"]
        non_zero = non_zero_list[-1]
        best_valid_acc = max(valid_acc_list)
        best_valid_auc = max(valid_auc_list)
        print(" *** Resume: [{}] Test Acc: {:.2f} Auc: {:.4f}"
              " at epoch: {} ***".format(ckpt_name, checkpoint["test_acc"] * 100,
                                        checkpoint["test_auc"], checkpoint["epoch"]))
    else:
        epoch = start_epoch = 0
        non_zero = 1
        train_loss_list = []
        train_acc_list = []
        train_auc_list = []
        valid_loss_list = []
        valid_acc_list = []
        valid_auc_list = []
        non_zero_list = []
        best_valid_acc = -1
        best_valid_auc = -1

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    n_epoch_wo_improvement = 0
    for epoch in range(start_epoch, args.max_epoch):
        if args.policy == "NoisyWideResNet":
            optimizer = torch.optim.SGD(param_list, lr=learning_rate(0.1, epoch),
                    momentum=0.9, weight_decay=5e-4)
        # train for one epoch
        train_loss, train_acc, train_auc, forward_time, backward_time, optim_time = \
           train(train_loader, n_cls, model, criterion, optimizer)
        # evaluate on validation set
        valid_loss, valid_acc, valid_auc = validate(valid_loader, n_cls, model, criterion)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_auc_list.append(train_auc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        valid_auc_list.append(valid_auc)
        non_zero = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                non_zero += param.abs().sign().sum().item()
        non_zero_list.append(non_zero)
        print('[{}] Train Acc: {:.2f}, Auc: {:.4f},'
              ' Valid Acc: {:.2f}, Auc: {:.4f},'
              ' log(Non_zero)={:.2f}'.format(ckpt_name.rstrip(".pth.tar"),
                                            train_acc*100, train_auc,
                                            valid_acc*100, valid_auc,
                                            np.log(non_zero)))
        is_best = valid_acc > best_valid_acc
        best_valid_acc = max(valid_acc, best_valid_acc)
        if is_best:
            n_epoch_wo_improvement = 0
            _, test_acc, test_auc = validate(test_loader, n_cls, model)
            state = {
                "act_fn": args.act_fn,
                "policy": args.policy,
                "noise_layer": args.noise_layer,
                "non_zero_list": non_zero_list,
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'train_auc_list': train_auc_list,
                'valid_loss_list': valid_loss_list,
                "valid_acc_list": valid_acc_list,
                "valid_auc_list": valid_auc_list,
                "test_acc": test_acc,
                "test_auc": test_auc,
                'optim_state_dict': optimizer.state_dict(),
                'forward_time': forward_time,
                'backward_time': backward_time,
                'optim_time': optim_time
            }
            save_checkpoint(state, args.save_dir, "best_" + ckpt_name)
        else:
            n_epoch_wo_improvement += 1

        if epoch > 0 and epoch % SAVE_EPOCH == 0:
            _, test_acc, test_auc = validate(test_loader, n_cls, model)
            state = {
                "act_fn": args.act_fn,
                "policy": args.policy,
                "noise_layer": args.noise_layer,
                "non_zero_list": non_zero_list,
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'train_auc_list': train_auc_list,
                'valid_loss_list': valid_loss_list,
                "valid_acc_list": valid_acc_list,
                "valid_auc_list": valid_auc_list,
                "test_acc": test_acc,
                'optim_state_dict': optimizer.state_dict(),
                'forward_time': forward_time,
                'backward_time': backward_time,
                'optim_time': optim_time
            }
            save_checkpoint(state, args.save_dir, "{}epoch_".format(epoch)+ckpt_name)
        if n_epoch_wo_improvement > EARLY_STOPPING_CRITERION :
            break
    _, test_acc, test_auc = validate(test_loader, n_cls, model)
    state = {
        #"model": args.model,
        "act_fn": args.act_fn,
        "policy": args.policy,
        "noise_layer": args.noise_layer,
        "non_zero_list": non_zero_list,
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'train_auc_list': train_auc_list,
        'valid_loss_list': valid_loss_list,
        "valid_acc_list": valid_acc_list,
        "valid_auc_list": valid_auc_list,
        "test_acc": test_acc,
        "test_auc": test_auc,
        'optim_state_dict': optimizer.state_dict(),
        'forward_time': forward_time,
        'backward_time': backward_time,
        'optim_time': optim_time
    }
    save_checkpoint(state, args.save_dir, ckpt_name)
    print('[{}] Test Accuracy: {:.2f}, Test Auc: {:.4f}, log(Non_zero)={:.2f}'.format(
        ckpt_name, test_acc*100, test_auc, np.log(non_zero)))


def train(train_loader, n_cls, model, criterion, optimizer):
    """Train for one epoch on the training set"""
    # switch to train mode
    model.train()
    loss_part = []
    acc_part = []
    scores = []
    targets = []
    forward_time = 0
    backward_time = 0
    optim_time = 0
    for i, (input_, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()
        # compute output
        forward_start = time.time()
        output = model(input_)
        loss = criterion(output, target)
        forward_time += (time.time()-forward_start)
        preds = output.max(dim=1)[1]
        preds_prob = output.softmax(dim=1).cpu().data.numpy()
        scores.append(preds_prob)
        targets.append(target.cpu().data.numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        backward_start = time.time()
        loss.backward()
        backward_time += (time.time()-backward_start)
        #for m in model.features.modules():
        #    if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad:
        #        print(torch.norm(m.weight.grad.data, p=2))
        optim_start = time.time()
        optimizer.step()
        optim_time += (time.time()-optim_start)

        # measure accuracy and record loss
        acc = (preds == target).sum().item() / preds.size(0)
        loss_part.append(loss.item())
        acc_part.append(acc)
    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets)
    targets = np.eye(n_cls)[targets.astype(int)]
    from sklearn.metrics import roc_auc_score
    binary = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) == 2:
            binary.append(i)
    #auc = roc_auc_score(targets[:,binary], scores[:, binary], 'macro')
    return np.mean(loss_part), np.mean(acc_part), 0, forward_time, backward_time, optim_time # auc

def validate(val_loader, n_cls, model, criterion=None):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()
    loss_part = []
    acc_part = []
    scores= []
    targets = []
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda(async=True)
                input_ = input_.cuda()
            # compute output
            output = model(input_)
            preds = output.max(dim=1)[1]
            preds_prob = output.softmax(dim=1).cpu().data.numpy()
            scores.append(preds_prob)
            targets.append(target.cpu().data.numpy())
            if criterion is not None:
                loss = criterion(output, target)
                loss = loss.item()
                loss_part.append(loss)
            else:
                loss_part.append(0)
            acc = (preds == target).sum().item() / preds.size(0)
            acc_part.append(acc)
    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets)
    targets = np.eye(n_cls)[targets.astype(int)]
    from sklearn.metrics import roc_auc_score
    binary = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) == 2:
            binary.append(i)
    #auc = roc_auc_score(targets[:, binary], scores[:, binary], 'macro')
    return np.mean(loss_part), np.mean(acc_part), 0# auc

def ROC_AUC(p, y):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc

def save_checkpoint(state, directory, name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name)
    torch.save(state, filename)

main()
