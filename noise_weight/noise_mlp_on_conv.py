import torch
from torchvision import datasets, transforms
from wide_resnet import WideResNet
from models_yki import NoisyMLP
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import copy
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--noise_layer", type=int, required=True, help="-1 means training whole networks")
parser.add_argument("--hdim", type=int, default=1)
parser.add_argument('--max_epoch', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument("--save_dir", type=str, default="ckpt")
parser.add_argument("--policy", type=str, default="NoisyMLPonWRN", choices=["NoisyMLPonWRN"])
parser.add_argument("--rand_seed", type=int, default=11)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
args = parser.parse_args()

CKPT_DIR = args.save_dir
ckpt = torch.load('ckpt/augment_cifar100_NoisyWideResNet_na_na_0_relu_0.pth.tar')
wrn = WideResNet(28, 10, 0.3, 100)
wrn.load_state_dict(ckpt["model_state_dict"])
device = torch.device('cuda')
wrn.to(device)
transform = transforms.Compose([transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
        0.225])])#,

test_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
        0.225])])#,
    #transforms.Lambda(lambda x: wrn.forward_conv(x.view(-1, 3, 32,
    #    32)).detach())])

train_set = datasets.CIFAR100(root='../data',train=True, download=True,
            transform=transform)
test_set = datasets.CIFAR100(root='../data',train=False, download=True,
            transform=test_transform)


train_loader =  torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size,
        shuffle=True, num_workers=4)
test_loader =  torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=4)
valid_loader = test_loader






def main():
    ckpt_name = "{}_{}_{}_{}.pth.tar".format('augment_cifar100', args.policy,
            args.noise_layer, args.hdim, args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    # create model
    model = NoisyMLP(input_dim=640, n_cls=100, noise_layer=args.noise_layer, times=args.hdim,
            dropout=0)
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
        #cudnn.benchmark = True

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

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    n_epoch_wo_improvement = 0
    for epoch in range(start_epoch, args.max_epoch):
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        valid_loss, valid_acc = validate(valid_loader, model, criterion)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        non_zero = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                non_zero += param.abs().sign().sum().item()
        non_zero_list.append(non_zero)
        print('[{}/ {}] [{}] Train Acc: {:.2f} Valid Acc: {:.2f}, log(Non_zero)={:.2f}'.format(epoch, args.max_epoch, ckpt_name.rstrip(".pth.tar"),
                                                                                      train_acc * 100,
                                                                                      valid_acc*100,
                                                                                      np.log(non_zero)))
        is_best = valid_acc > best_valid_acc
        best_valid_acc = max(valid_acc, best_valid_acc)
        if is_best:
            n_epoch_wo_improvement = 0
            _, test_acc = validate(test_loader, model)
            state = {
                "model": args.model,
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
            _, test_acc = validate(test_loader, model)
            state = {
                "model": args.model,
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
    _, test_acc = validate(test_loader, model)
    state = {
        "model": args.model,
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

def train(train_loader, model, criterion, optimizer):
    """Train for one epoch on the training set"""
    # switch to train mode
    model.train()
    loss_part = []
    acc_part = []
    for X, y in train_loader:
        with torch.no_grad():
            input_ = wrn.forward_conv(X.to(device)).detach()
            target = y.to(device)

        # compute output
        output = model(input_)
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
    return np.mean(loss_part), np.mean(acc_part)

def validate(val_loader, model, criterion=None):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()
    loss_part = []
    acc_part = []
    with torch.no_grad():
        for X, y in train_loader:
            input_ = wrn.forward_conv(X.to(device)).detach()
            target = y.to(device)
            # compute output
            output = model(input_)
            preds = output.max(dim=1)[1]
            if criterion is not None:
                loss = criterion(output, target)
                loss = loss.item()
                loss_part.append(loss)
            else:
                loss_part.append(0)
            acc = (preds == target).sum().item() / preds.size(0)
            acc_part.append(acc)
    return np.mean(loss_part), np.mean(acc_part)

def save_checkpoint(state, directory, name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name)
    torch.save(state, filename)

main()
