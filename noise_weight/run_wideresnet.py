import time as time_
import datetime
from time import time
import random
import numpy as np
from multiprocessing import Pool
import os
import multiprocessing
from multiprocessing import Pool
import argparse
import subprocess

manager = multiprocessing.Manager()
GPUqueue = manager.Queue()
for i in range(4):
    GPUqueue.put(i)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    choices=["cifar10","cifar100"])
#parser.add_argument("--augment", action='store_true', default=False)
args = parser.parse_args()


def launch_experiment(args):
    subprocess.run(args=['python3', 'train_cnn.py', '--augment',
	'--max_epoch', '200',
        '--dataset', str(args[1]),
        '--rand_seed', str(args[2]), '--act_fn', str(args[3]),
        '--noise_layer', str(args[4]), '--policy', str(args[5]),
        '--optim', args[6]])
    launch_experiment.GPUq.put(int(os.environ['CUDA_VISIBLE_DEVICES']))
    return args

def distribute_gpu(q):
    launch_experiment.GPUq = q
    num = q.get()
    print("process id = {0}: using gpu {1}".format(os.getpid(), num))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num)

def main():
    i = 0
    n_process = 4
    rand_seed_list = range(1)
    act_fn = "relu"
    noise_layer_list = range(4)
    args_list = []
    for rand_seed in rand_seed_list:
        for noise_layer in noise_layer_list:
            args_list.append([i % 4, args.dataset,
                rand_seed, act_fn, noise_layer, "NoisyWideResNet", "adam"])
            i += 1
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process, initializer=distribute_gpu,
            initargs=(GPUqueue,), maxtasksperchild=1)
    pool.map(launch_experiment, args_list)

main()
#os.system("scp -r ckpt/ yki@143.248.57.168:/hdd01/MLILAB/research/noisynet/experiment/NoisyMLP/")
