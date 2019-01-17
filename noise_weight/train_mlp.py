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

parser = argparse.ArgumentParser(description='L0-neuron MLP-300-100 Training')
parser.add_argument('--max_epoch', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument("--act_fn", type=str, default="relu")
parser.add_argument("--noise_layer", type=int, default=-1)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--model', default='MLP-300', type=str,
                    help='name of experiment')
parser.add_argument("--save_dir", type=str, default="ckpt")
parser.add_argument("--policy", type=str, default="NoisyMLP")
parser.add_argument("--rand_seed", type=int, default=11)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", choices=['wine', 'mnist', 'abalone'], default='wine')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
CKPT_DIR = "ckpt"
DATA_DIR = "../data"
EARLY_STOPPING_CRITERION = args.max_epoch  # If there is no learning improvement withing the consecutive N epochs, stopping
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
SAVE_EPOCH = 100


def main():
    ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, args.model, args.policy, args.noise_layer, args.act_fn, args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
    # create model
    token = args.model.split("-")
    layer_dims = tuple([int(elem) for elem in token[1:]])
    print(len(layer_dims), layer_dims, args.policy,args.noise_layer, args.act_fn)
    # data
    if args.dataset == 'wine':
        from dataloader import WhiteWine
        train_set = WhiteWine(args.rand_seed, train=True, test_ratio=0.3)
        valid_set = WhiteWine(args.rand_seed, train=True, test_ratio=0.3)
        #assert np.all(train_set[1][0] == valid_set[1][0])
        test_set = WhiteWine(args.rand_seed, train=False, test_ratio=0.3)
        input_dim = train_set.X.shape[1]
        print(train_set.weight)
        #input()
        n_cls = 7
    elif args.dataset == 'abalone':
        from dataloader import Abalone
        train_set = Abalone(args.rand_seed, train=True, test_ratio=0.3)
        valid_set = Abalone(args.rand_seed, train=True, test_ratio=0.3)
        #assert np.all(train_set[1][0] == valid_set[1][0])
        test_set = Abalone(args.rand_seed, train=False, test_ratio=0.3)
        input_dim = train_set.X.shape[1]
        n_cls = 29
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

    model = NoisyMLP(input_dim=input_dim, n_cls=n_cls, layer_dims=layer_dims,
            activation_fn=args.act_fn, noise_layer=args.noise_layer)
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
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
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
        valid_loss_list = checkpoint["valid_loss_list"]
        valid_acc_list = checkpoint["valid_acc_list"]
        non_zero_list = checkpoint["non_zero_list"]
        non_zero = non_zero_list[-1]
        best_valid_acc = max(valid_acc_list)
        print(" *** Resume: [{}] Test Acc: {:.2f} at epoch: {} ***".format(ckpt_name, checkpoint["test_acc"] * 100,
                                                                           checkpoint["epoch"]))
    elif os.path.exists(os.path.join(CKPT_DIR, "best_" + ckpt_name)):
        checkpoint = torch.load(os.path.join(CKPT_DIR, "best_" + ckpt_name))
        epoch = start_epoch = checkpoint['epoch']
        test_acc = checkpoint['test_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        train_loss_list = checkpoint["train_loss_list"]
        train_acc_list = checkpoint["train_acc_list"]
        valid_loss_list = checkpoint["valid_loss_list"]
        valid_acc_list = checkpoint["valid_acc_list"]
        non_zero_list = checkpoint["non_zero_list"]
        best_valid_acc = max(valid_acc_list)
        print(" *** Resume: [{}] Test Acc: {:.2f}, epoch: {} ***".format("best_" + ckpt_name,
                                                                         checkpoint["test_acc"] * 100,
                                                                         checkpoint["epoch"]))
    else:
        start_epoch = 0
        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        non_zero_list = []
        best_valid_acc = -1

    criterion = nn.CrossEntropyLoss()#weight=train_set.weight.float().cuda())
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    n_epoch_wo_improvement = 0
    for epoch in range(start_epoch, args.max_epoch):
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer,
                n_cls)
        # evaluate on validation set
        valid_loss, valid_acc = validate(valid_loader, model, n_cls, criterion)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        non_zero = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                non_zero += param.abs().sign().sum().item()
        non_zero_list.append(non_zero)
        print('[{}] Train Acc: {:.2f} Valid Acc: {:.2f}, log(Non_zero)={:.2f}'.format(ckpt_name.rstrip(".pth.tar"),
                                                                                      train_acc * 100,
                                                                                      valid_acc*100,
                                                                                      np.log(non_zero)))
        is_best = valid_acc > best_valid_acc
        best_valid_acc = max(valid_acc, best_valid_acc)
        if is_best:
            n_epoch_wo_improvement = 0
            _, test_acc = validate(test_loader, model, n_cls)
            state = {
                "model": args.model,
                "act_fn": args.act_fn,
                "noise_layer" : args.noise_layer,
                "non_zero_list": non_zero_list,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'valid_loss_list': valid_loss_list,
                "valid_acc_list": valid_acc_list,
                "test_acc": test_acc,
                'optim_state_dict': optimizer.state_dict()
            }
            save_checkpoint(state, args.save_dir, "best_" + ckpt_name)
        else:
            n_epoch_wo_improvement += 1

        if epoch > 0 and epoch % SAVE_EPOCH == 0:
            _, test_acc = validate(test_loader, model, n_cls)
            state = {
                "model": args.model,
                "act_fn": args.act_fn,
                "noise_layer" : args.noise_layer,
                "non_zero_list": non_zero_list,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'valid_loss_list': valid_loss_list,
                "valid_acc_list": valid_acc_list,
                "test_acc": test_acc,
                'optim_state_dict': optimizer.state_dict()
            }
            save_checkpoint(state, args.save_dir, "{}epoch_".format(epoch)+ckpt_name)
        if n_epoch_wo_improvement > EARLY_STOPPING_CRITERION :
            break
    _, test_acc = validate(test_loader, model, n_cls)
    state = {
        "model": args.model,
        "act_fn": args.act_fn,
        "noise_layer" : args.noise_layer,
        "non_zero_list": non_zero_list,
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'valid_loss_list': valid_loss_list,
        "valid_acc_list": valid_acc_list,
        "test_acc": test_acc,
        'optim_state_dict': optimizer.state_dict()
    }
    save_checkpoint(state, args.save_dir, ckpt_name)
    print('[{}] Test Accuracy: {:.2f}, log(Non_zero)={:.2f}'.format(ckpt_name, test_acc * 100, np.log(non_zero)))

def train(train_loader, model, criterion, optimizer, n_cls, metric='auc'):
    """Train for one epoch on the training set"""
    # switch to train mode
    model.train()
    loss_part = []
    acc_part = []
    scores = []
    targets = []
    for i, (input_, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()

        # compute output
        output = model(input_)
        pred_prob = output.cpu().softmax(dim=1).data.numpy()
        scores.append(pred_prob)
        targets.append(target.cpu().data.numpy())
        preds = output.max(dim=1)[1]
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = (preds == target).sum().item() / preds.size(0)
        loss_part.append(loss.item())
        acc_part.append(acc)
    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets)
    targets = np.eye(n_cls)[targets.astype(int)]

    from sklearn.metrics import roc_auc_score

    binary  = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) == 2:
            binary.append(i)
    auc = roc_auc_score(targets[:, binary], scores[:, binary], 'macro')
    if metric == 'acc':
        return np.mean(loss_part), np.mean(acc_part)
    elif metric == 'auc':
        return np.mean(loss_part), auc# np.mean(acc_part)

def validate(val_loader, model, n_cls, criterion=None, metric='auc'):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()
    loss_part = []
    acc_part = []

    scores = []
    targets = []
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda(async=True)
                input_ = input_.cuda()
            # compute output
            output = model(input_)
            pred_prob = output.cpu().softmax(dim=1).data.numpy()
            scores.append(pred_prob)
            targets.append(target.cpu().data.numpy())
            preds = output.max(dim=1)[1]
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

        binary  = []
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) == 2:
                binary.append(i)
        auc = roc_auc_score(targets[:, binary], scores[:, binary], 'macro')
    if metric == 'acc':
        return np.mean(loss_part), np.mean(acc_part)
    elif metric == 'auc':
        return np.mean(loss_part), auc # np.mean(acc_part)

def save_checkpoint(state, directory, name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name)
    torch.save(state, filename)

main()
