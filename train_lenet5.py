import argparse
import shutil
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from models import L0LeNet5
from utils import save_checkpoint
#from dataloaders import mnist
from data_loader import pMNIST
from utils import AverageMeter, accuracy


parser = argparse.ArgumentParser(description='PyTorch LeNet5 Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='L0LeNet5-20-50-500', type=str,
                    help='name of experiment')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument("--noise", action="store_true", default=True,
                    help="Random Noise on weight matrix")
parser.add_argument('--beta_ema', type=float, default=0.0)
parser.add_argument('--lambas', nargs='*', type=float, default=[0.1]*4)
parser.add_argument('--local_rep', action='store_true')
parser.add_argument('--temp', type=float, default=2./3.)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument("--policy", type=str, default="L0")
parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--rand_seed", type=int, default=1)
parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist",
    "cifar10", "cifar100"])
parser.set_defaults(tensorboard=True)

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
CKPT_DIR = "ckpt"
EARLY_STOPPING_CRITERION = 50  # If there is no learning improvement withing the consecutive N epochs, stopping
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.rand_seed)
ZERO_THRESHOLD = 1e-5
#parser.set_defaults(tensorboard=False)
writer = None
total_steps = 0
exp_flops, exp_l0 = [], []


def main():
    global args, best_prec1, writer, total_steps, exp_flops, exp_l0
    args = parser.parse_args()
    log_dir_net = args.name
    print('model:', args.name)

    token = args.name.split("-")
    args.conv_dims = list(map(int, token[1:3]))
    if args.dataset == 'mnist':
        args.fc_dims = int(token[3])
    else:
        args.fc_dims = list(map(int, token[3:]))

    augment = "Aug" if args.dataset != 'mnist' else ""
    ckpt_name = ("{}_{}_{}_policy_{}_{:.2f}_noise_{:.3f}" + "_{:.2e}" * len(args.lambas) + \
            "_{}_{:.2f}_epochs_{}.pth.tar").format(
        args.name, args.dataset + augment, args.policy, args.rand_seed, args.sparsity, args.beta_ema,
        *args.lambas, args.local_rep, args.temp, args.epochs
    )
    if args.tensorboard:
        # used for logging to TensorBoard
        from tensorboardX import SummaryWriter
        directory = ckpt_name + '/logs'
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        writer = SummaryWriter(directory)

    # Data loading code
    #print('[0, 1] normalization of input')
    #train_loader, val_loader, num_classes = mnist(args.batch_size, pm=False)

    num_classes=10
    input_size = (1, 28, 28) if args.dataset == 'mnist' else (3, 32, 32)
    # create model
    model = L0LeNet5(num_classes, input_size=input_size,
            conv_dims=args.conv_dims, fc_dims=args.fc_dims, N=60000,
                     weight_decay=args.weight_decay, lambas=args.lambas, local_rep=args.local_rep,
                     temperature=args.temp, beta_ema=args.beta_ema)

    param_list, name_list, noise_param_list, noise_name_list = [], [], [], []
    for name, param in model.named_parameters():
        if "random" not in name:
            param_list.append(param)
            name_list.append(name)
        else:
            # Adjust the sparsity of random noise parameters
            if args.sparsity > 0:
                param.data.mul_(torch.bernoulli(torch.ones_like(param) * (1 - args.sparsity)))
            if args.verbose:
                print("[{}] size: {}, # Non-zeros ratio= {:.2f}".format(name, param.size(),
                                                                        param.data.sign().abs().sum().item() / np.prod(
                                                                            param.size()) * 100))
            noise_param_list.append(param)
            noise_name_list.append(name)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # optionally resume from a checkpoint
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    nonzero_list = []
    if args.resume:
        path = os.path.join(CKPT_DIR, ckpt_name)
        if os.path.exists(path):
            print("=> loading checkpoint '{}'".format(ckpt_name))
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch']
            test_acc = checkpoint['test_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            total_steps = checkpoint['total_steps']
            exp_flops = checkpoint['exp_flops']
            exp_l0 = checkpoint['exp_l0']
            train_loss_list = checkpoint["train_loss_list"]
            train_acc_list = checkpoint["train_acc_list"]
            valid_loss_list = checkpoint["valid_loss_list"]
            valid_acc_list = checkpoint["valid_acc_list"]
            nonzero_list = checkpoint["nonzero_list"]
            print(" *** Resume: [{}] Test Acc: {:.2f}, epoch: {} ***".format(ckpt_name, checkpoint["test_acc"]*100, checkpoint["epoch"]))
            if checkpoint['beta_ema'] > 0:
                #model.beta_ema = checkpoint['beta_ema']
                model.avg_param = checkpoint['avg_params']
                model.steps_ema = checkpoint['steps_ema']
        else:
            total_steps, exp_flops, exp_l0 = 0, [], []

    loglike = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loglike = loglike.cuda()

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        total_loss = loss + model.regularization()
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    if args.dataset.startswith("cifar"):
        from torchvision import transforms, datasets
        DATA_DIR = './data/cifar10'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        n_cls = 10 if args.dataset == 'cifar10' else 100
        dset_string = 'datasets.CIFAR10' if args.dataset == 'cifar10' else 'datasets.CIFAR100'
        train_tfms = [transforms.ToTensor(), normalize]
        test_set = eval(dset_string)(root=DATA_DIR, train=False,
                                      transform=transforms.Compose(train_tfms), download=True)
        valid_set = eval(dset_string)(root=DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)
        if augment:
            train_tfms = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip()] + train_tfms
        train_set = eval(dset_string)(root=DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)
    else:
        train_set = pMNIST(flat=False)
        valid_set = pMNIST(flat=False)
        test_set = pMNIST(train=False, flat=False)

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

    best_valid_acc = -1
    n_epoch_wo_improvement = 0
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        before_train_randomparam = [param.data.clone() for param in noise_param_list]
        train_loss, train_acc = train(train_loader, model, loss_function, optimizer, epoch)
        after_train_randomparam = [param.data.clone() for param in noise_param_list]
        # random weight constant test
        for b, a in zip(before_train_randomparam, after_train_randomparam):
            assert torch.all(b == a)
        # evaluate on validation set
        valid_loss, valid_acc = validate(valid_loader, model, loss_function, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        features, architecture = model.compute_params()
        non_zero = np.sum(features)
        nonzero_list.append(non_zero)

        # remember best prec@1 and save checkpoint
        is_best = valid_acc > best_valid_acc
        best_valid_acc = max(valid_acc, best_valid_acc)

        if is_best or epoch > 5:
            n_epoch_wo_improvement = 0
            test_acc = test(test_loader, model, loss_function, epoch)
            state = {
                "model": args.name,
                "pruned_model": model.compute_params(),
                'epoch': epoch + 1,
                "noise": args.noise,
                'model_state_dict': model.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'valid_loss_list': valid_loss_list,
                "valid_acc_list": valid_acc_list,
                "nonzero_list": nonzero_list,
                "test_acc": test_acc,
                'optim_state_dict': optimizer.state_dict(),
                'total_steps': total_steps,
                'exp_flops': exp_flops,
                'exp_l0': exp_l0
            }
            if model.beta_ema > 0:
                state['avg_params'] = model.avg_param
                state['steps_ema'] = model.steps_ema
            save_checkpoint(state, is_best, ckpt_name, epoch + 1)
        else:
            n_epoch_wo_improvement += 1

    features, architecture = model.compute_params()
    non_zero = np.sum(features)
    print(non_zero)

    total = sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])
    print('[{}] Test Accuracy: {:.2f}, Non_zero={}'.format(ckpt_name, test_acc*100, non_zero))
    if args.tensorboard:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    global total_steps, exp_flops, exp_l0, args, writer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    loss_part = []
    acc_part = []
    for i, (input_, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        total_steps += 1
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()
        #input_var = torch.autograd.Variable(input_)
        #target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_)
        preds = output.max(dim=1)[1]
        loss = criterion(output, target, model)

        # measure accuracy and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))[0]
        prec1 = (preds == target).sum().item() / preds.size(0)
        losses.update(loss.item(), input_.size(0))
        top1.update(100 - prec1*100, input_.size(0))
        loss_part.append(loss.item())
        acc_part.append(prec1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp the parameters
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        e_fl, e_l0 = model.get_exp_flops_l0() if not args.multi_gpu else \
            model.module.get_exp_flops_l0()
        exp_flops.append(e_fl)
        exp_l0.append(e_l0)
        if writer is not None:
            writer.add_scalar('stats_comp/exp_flops', e_fl, total_steps)
            writer.add_scalar('stats_comp/exp_l0', e_l0, total_steps)

        if not args.multi_gpu:
            if model.beta_ema > 0.:
                model.update_ema()
        else:
            if model.module.beta_ema > 0.:
                model.module.update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # input()
        if (i + 1) % args.print_freq == 0 and args.verbose:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', np.mean(acc_part), epoch)
        writer.add_scalar('train/err', top1.avg, epoch)

    return np.mean(loss_part), np.mean(acc_part)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    global args, writer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if not args.multi_gpu:
        if model.beta_ema > 0:
            old_params = model.get_params()
            model.load_ema_params()
    else:
        if model.module.beta_ema > 0:
            old_params = model.module.get_params()
            model.module.load_ema_params()
    end = time.time()
    loss_part = []
    acc_part = []
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda(async=True)
                input_ = input_.cuda()
            #input_var = torch.autograd.Variable(input_, volatile=True)
            #target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_)
            preds = output.max(dim=1)[1]
            loss = criterion(output, target, model)

            # measure accuracy and record loss
            #prec1 = accuracy(output.data, target, topk=(1,))[0]
            prec1 = (preds == target).sum().item() / preds.size(0)
            losses.update(loss.item(), input_.size(0))
            top1.update(100 - prec1*100, input_.size(0))
            loss_part.append(loss.item())
            acc_part.append(prec1)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0 and args.verbose:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
    if args.verbose:
        print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))
    if not args.multi_gpu:
        if model.beta_ema > 0:
            model.load_params(old_params)
    else:
        if model.module.beta_ema > 0:
            model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        writer.add_scalar('val/acc', np.mean(acc_part))
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)

    return np.mean(loss_part), np.mean(acc_part)

def test(test_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    global args, writer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if not args.multi_gpu:
        if model.beta_ema > 0:
            old_params = model.get_params()
            model.load_ema_params()
    else:
        if model.module.beta_ema > 0:
            old_params = model.module.get_params()
            model.module.load_ema_params()

    end = time.time()
    acc_part = []
    with torch.no_grad():
        for i, (input_, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                target = target.cuda(async=True)
                input_ = input_.cuda()
            # compute output
            output = model(input_)
            preds = output.max(dim=1)[1]

            # measure accuracy and record loss
            # prec1 = accuracy(output.item(), target, topk=(1,))[0]
            prec1 = (preds == target).sum().item() / preds.size(0)
            top1.update(100 - prec1*100, input_.size(0))
            acc_part.append(prec1)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.verbose:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    if args.verbose:
        print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))
    if not args.multi_gpu:
        if model.beta_ema > 0:
            model.load_params(old_params)
    else:
        if model.module.beta_ema > 0:
            model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('test/loss', losses.avg, epoch)
        writer.add_scalar('test/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return np.mean(acc_part)


if __name__ == '__main__':
    main()
