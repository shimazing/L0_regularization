import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_auc_score
from models import *
from torchvision import datasets, transforms
import copy
import argparse
import os
import numpy as np
from dataloader import *
import time
from operator import itemgetter


parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument("--act_fn", type=str, default="relu")
parser.add_argument("--hdim", type=int, required=True)
parser.add_argument("--nlayer", type=int, required=True)
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument("--save_dir", type=str, default="ckpt")
parser.add_argument("--policy", type=str, required=True,
        choices=['IntermediateNoisyMLP',
                 'AlternatingNoisyMLP',
                 'IncomingNoisyMLP',
                 'OutgoingNoisyMLP'
                ])
parser.add_argument("--noise_layer", type=int, required=True,
                    help="-1 means training whole networks")
parser.add_argument("--batchnorm", type=str, default='none', choices=['none',
    'before', 'after'])
parser.add_argument("--input_drop", type=float, default=0.)
parser.add_argument("--hidden_drop", type=float, default=0.)
parser.add_argument("--rand_seed", type=int, default=11)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--dataset", type=str, required=True,
                    choices=["letter", "dna", "satimage", "ijcnn1", "mnist", "svhn",
                             "whitewine", "redwine", "abalone", "fashionmnist",
                             "cifar10", "cifar100", "preprocessed-cifar10",
                             "preprocessed-cifar100"
                             ])

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
cudnn.benchmark = args.cuda
CKPT_DIR = "ckpt"
DATA_DIR = "../data"
EARLY_STOPPING_CRITERION = args.max_epoch  # If there is no learning improvement withing the consecutive N epochs, stopping
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
SAVE_EPOCH = 100


def main():
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    ckpt_name = gen_ckpt_name(vars(args))
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
    #-------------------------------------------------------------------------------------------------------------------
    # Load dataset
    train_loader, valid_loader, test_loader, n_cls, n_features = \
        load_data(args.dataset, args, kwargs, args.rand_seed)
    #------------------------------------------------------------------------------------------------------------------
    # create model
    layer_dims = tuple([args.hdim]*args.nlayer)
    print(args.dataset, len(layer_dims), layer_dims, args.policy, args.noise_layer)
    model = eval(args.policy)(input_dim=n_features, n_cls=n_cls,
                              layer_dims=layer_dims,
                              activation_fn=args.act_fn,
                              noise_layer=args.noise_layer,
                              batchnorm=args.batchnorm,
                              input_drop=args.input_drop,
                              hidden_drop=args.hidden_drop).to(device)
    param_list, name_list, noise_param_list, noise_name_list = [], [], [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            name_list.append(name)
        else:
            noise_param_list.append(param)
            noise_name_list.append(name)
    optimizer = torch.optim.Adam(param_list, args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    #------------------------------------------------------------------------------------------------------------------
    # optionally resume from a checkpoint
    items = ['loss', 'acc', 'auc']
    if os.path.exists(os.path.join(CKPT_DIR, ckpt_name)) or \
            os.path.exists(os.path.join(CKPT_DIR, "best_" + ckpt_name)):

        if os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
            checkpoint = torch.load(os.path.join(CKPT_DIR, ckpt_name))
        elif os.path.exists(os.path.join(CKPT_DIR, "best_" + ckpt_name)):
            checkpoint = torch.load(os.path.join(CKPT_DIR, "best_" + ckpt_name))
        epoch = start_epoch = checkpoint['epoch']
        test_acc = checkpoint['test_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        train_loss_list, train_acc_list, train_auc_list = itemgetter(*["train_{}_list".format(it) for it in items])(checkpoint)
        valid_loss_list, valid_acc_list, valid_auc_list = itemgetter(*["valid_{}_list".format(it) for it in items])(checkpoint)
        non_zero = checkpoint["non_zero"]
        forward_time_list = checkpoint["forward_time_list"]
        backward_time_list = checkpoint["backward_time_list"]
        step_time_list = checkpoint["step_time_list"]
        best_valid_acc = max(valid_acc_list)
        best_valid_auc = max(valid_auc_list)
        print(" *** Resume: [{}] Test Acc: {:.2f} Auc: {:.4f}"
              " at epoch: {} ***".format(ckpt_name, checkpoint["test_acc"]*100,
                                        checkpoint["test_auc"], checkpoint["epoch"]))
    else:
        epoch = start_epoch = 0
        non_zero = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                non_zero += param.abs().sign().sum().item()
        train_loss_list, train_acc_list, train_auc_list = [], [], []
        valid_loss_list, valid_acc_list, valid_auc_list = [], [], []
        forward_time_list, backward_time_list, step_time_list = [], [], []
        best_valid_acc = -1
        best_valid_auc = -1


    def make_state():
        _, test_acc, test_auc = validate(test_loader, n_cls, model,
                device=device)
        state = {
            "act_fn": args.act_fn,
            "policy": args.policy,
            "noise_layer": args.noise_layer,
            "non_zero": non_zero,
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'train_loss_list': train_loss_list,
            'train_acc_list': train_acc_list,
            'train_auc_list': train_auc_list,
            'valid_loss_list': valid_loss_list,
            "valid_acc_list": valid_acc_list,
            "valid_auc_list": valid_auc_list,
            "forward_time_list": forward_time_list,
            "backward_time_list": backward_time_list,
            "step_time_list": step_time_list,
            "test_acc": test_acc,
            "test_auc": test_auc,
            'optim_state_dict': optimizer.state_dict()
        }
        return state, test_acc, test_auc
    #------------------------------------------------------------------------------------------------------------------
    # start training
    n_epoch_wo_improvement = 0
    for epoch in range(start_epoch, args.max_epoch):
        train(train_loader, n_cls, model, criterion, optimizer, device,
              forward_time_list, backward_time_list, step_time_list,
              train_loss_list, train_acc_list, train_auc_list)
        validate(valid_loader, n_cls, model, criterion, device,
                valid_loss_list, valid_acc_list, valid_auc_list)
        print('[{},{}/{}] Train Acc: {:.2f}, Valid Acc: {:.2f}, log(Non_zero)={:.2f}'\
                .format(ckpt_name.rstrip(".pth.tar"), epoch, args.max_epoch,
                        train_acc_list[-1]*100, valid_acc_list[-1]*100, np.log(non_zero)))
        is_best = valid_auc_list[-1] > best_valid_auc
        best_valid_auc = max(valid_auc_list[-1], best_valid_auc)
        if is_best or (epoch > 0 and epoch % SAVE_EPOCH == 0):
            state, test_acc, test_auc = make_state()
            if is_best:
                n_epoch_wo_improvement = 0
                torch.save(state, os.path.join(args.save_dir, "best_" +
                    ckpt_name))
            if epoch % SAVE_EPOCH == 0:
                torch.save(state, os.path.join(args.save_dir,
                    "{}epoch_".format(epoch) + ckpt_name))
        else:
            if not is_best:
                n_epoch_wo_improvement += 1
        if n_epoch_wo_improvement > EARLY_STOPPING_CRITERION :
            break
    state, test_acc, test_auc = make_state()
    torch.save(state, os.path.join(args.save_dir, ckpt_name))
    print('[{}] Test Accuracy: {:.2f}, log(Non_zero)={:.2f}'.format(
        ckpt_name, test_acc*100, np.log(non_zero)))


def train(train_loader, n_cls, model, criterion, optimizer, device,
        forward_time_list, backward_time_list, step_time_list,
        loss_list, acc_list, auc_list):
    """Train for one epoch on the training set"""
    # switch to train mode
    model.train()
    loss_part, acc_part = [], []
    scores, targets = [], []
    forward_time, backward_time, optim_step_time = [], [], []
    for i, (input_, target) in enumerate(train_loader):
        target = target.to(device)
        input_ = input_.to(device)
        # compute output
        # ---------------------------------------------------------------------------------------------------------
        # Forward
        start_time = time.time()
        output = model(input_)
        end_time = time.time()
        forward_time.append(end_time - start_time)
        # ---------------------------------------------------------------------------------------------------------
        preds = output.max(dim=1)[1]
        preds_prob = output.softmax(dim=1).cpu().data.numpy()
        scores.append(preds_prob)
        targets.append(target.cpu().data.numpy())
        loss = criterion(output, target)

        # compute gradient and do SGD step
        # ---------------------------------------------------------------------------------------------------------
        # Backward
        optimizer.zero_grad()
        start_time = time.time()
        loss.backward()
        end_time = time.time()
        backward_time.append(end_time - start_time)

        start_time = time.time()
        optimizer.step()
        end_time = time.time()
        optim_step_time.append(end_time - start_time)
        # ----------------------------------------------------------------------------------------------------------
        # measure accuracy and record loss
        acc = (preds == target).sum().item() / preds.size(0)
        loss_part.append(loss.item())
        acc_part.append(acc)
    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets)
    targets = np.eye(n_cls)[targets.astype(int)]
    binary = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) == 2:
            binary.append(i)
    auc = roc_auc_score(targets[:,binary], scores[:, binary], 'macro')

    forward_time, backward_time, optim_step_time = np.sum([forward_time, backward_time, optim_step_time], axis=-1)
    loss, acc = np.mean([loss_part, acc_part], axis=-1)
    forward_time_list.append(forward_time)
    backward_time_list.append(backward_time)
    step_time_list.append(optim_step_time)
    loss_list.append(loss)
    acc_list.append(acc)
    auc_list.append(auc)
    return loss, acc, auc, forward_time, backward_time, optim_step_time


def validate(val_loader, n_cls, model, criterion=None, device=None,
        loss_list=[], acc_list=[], auc_list=[]):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()
    loss_part = []
    acc_part = []
    scores= []
    targets = []
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            target = target.to(device)
            input_ = input_.to(device)
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
    binary = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) == 2:
            binary.append(i)
    auc = roc_auc_score(targets[:, binary], scores[:, binary], 'macro')
    loss, acc = np.mean([loss_part, acc_part], axis=-1)
    loss_list.append(loss)
    acc_list.append(acc)
    auc_list.append(auc)
    return loss, acc, auc


main()
