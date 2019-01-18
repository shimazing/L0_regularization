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

def run_exp(args):
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 train_cnn.py --dataset {1} --hdim {2} \
        --rand_seed {3} --act_fn {4} --noise_layer {5} --policy {6} ".format(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    print("[{}] {}".format(
        datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
        cmd
        ))
    time_.sleep(0.1)
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    os.system(cmd)

def gen_model_name_list(nhid_list, n_layer):
    model_name_list = []
    for nhid in nhid_list:
        model_name_list.append("MLP"+"-{}".format(nhid)*n_layer)
    return model_name_list

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    choices=["cifar10","cifar100"])
#parser.add_argument("--augment", action='store_true', default=False)
args = parser.parse_args()


def launch_experiment(args):
    subprocess.run(args=['python3', 'train_cnn.py',
        '--dataset', str(args[1]), '--hdim', str(args[2]),
        '--rand_seed', str(args[3]), '--act_fn', str(args[4]),
        '--noise_layer', str(args[5]), '--policy', str(args[6])])
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
    hdim_list = [512]#, 4096] #[2048, 1024]
    rand_seed_list = range(2)
    act_fn = "relu"
    noise_layer_list = range(11)
    args_list = []
    for hdim in hdim_list:
        for rand_seed in rand_seed_list:
            for noise_layer in noise_layer_list:
                args_list.append([i % 4, args.dataset, hdim,
                    rand_seed, act_fn, noise_layer, "NoisyVgg16"])
                i += 1
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process, initializer=distribute_gpu,
            initargs=(GPUqueue,), maxtasksperchild=1)
    pool.map(launch_experiment, args_list)

main()
#os.system("scp -r ckpt/ yki@143.248.57.168:/hdd01/MLILAB/research/noisynet/experiment/NoisyMLP/")
