from time import time
import random
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import argparse
import subprocess

manager = multiprocessing.Manager()
GPUqueue = manager.Queue()
processes_per_gpu = 10
for i in range(4):
    for _ in range(processes_per_gpu):
        GPUqueue.put(i)


def run_exp(args):
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 train_mlp.py --dataset {1} --rand_seed {2}"\
          " --noise_layer {3} --nlayer {4} --hdim {5} --policy {6} --act_fn {7}".format(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
    time_.sleep(0.1)
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    os.system(cmd)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
RAND_SEED_LIST = range(5)

args_list = []

def launch_experiment(args):
    args = [str(arg) for arg in args]
    subprocess.run(args=['python3', 'train_mlp.py',
        '--max_epoch', '200',
        '--dataset', args[0],
        '--rand_seed', args[1],
        '--noise_layer', args[2],
        '--policy', args[3],
        '--nlayer', args[4],
        '--hdim', args[5],
        '--input_drop', args[6],
        '--hidden_drop', args[7],
        '--batchnorm', args[8]])
    launch_experiment.GPUq.put(int(os.environ['CUDA_VISIBLE_DEVICES']))
    return args

def distribute_gpu(q):
    launch_experiment.GPUq = q
    num = q.get()
    print("process id = {0}: using gpu {1}".format(os.getpid(), num))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num)

def main():
    n_process = 4 * processes_per_gpu
    hdim_list = [10, 50, 100, 500]
    nlayer_list = [6,8,10]
    learnt_layer_list = [4,2]
    input_drop_list = [0, 0.2]
    hidden_drop_list = [0, 0.2, 0.5]
    batchnorm_list = ['none', 'before', 'after']

    for learnt_layer in learnt_layer_list:
        if learnt_layer == 2:
            frozen_train(hdim_list, [3, 4], learnt_layer,
                hidden_drop_list, input_drop_list, batchnorm_list)
        else:
            frozen_train(hdim_list, nlayer_list, learnt_layer,
                    hidden_drop_list, input_drop_list, batchnorm_list)
        fully_train(hdim_list, learnt_layer,
                hidden_drop_list, input_drop_list, batchnorm_list)
    print("# total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process,initializer=distribute_gpu,
            initargs=(GPUqueue,), maxtasksperchild=1)
    pool.map(launch_experiment, args_list)


def frozen_train(hdim_list, nlayer_list, learnt_layer,
        hidden_drop_list, input_drop_list, batchnorm_list):
    policy_list = ["IntermediateNoisyMLP", "IncomingNoisyMLP",
            "OutgoingNoisyMLP", "AlternatingNoisyMLP"]
    for policy in policy_list:
        for nlayer in nlayer_list:
            if policy == "IncomingNoisyMLP":
                noise_layer_list = [nlayer - learnt_layer] # Here, 0 means fully training
            elif policy == "OutgoingNoisyMLP":
                noise_layer_list = [learnt_layer]
            elif policy == "IntermediateNoisyMLP":
                assert learnt_layer % 2 == 0
                freezing_start = int(learnt_layer/2)
                freezing_end = nlayer - int(learnt_layer/2)
                noise_layer_list = ["{}{}".format(freezing_start, freezing_end)]
            elif policy == "AlternatingNoisyMLP":
                noise_layer_list = [np.ceil(nlayer/learnt_layer).astype(int)]
            else:
                raise ValueError
            for hdim in hdim_list:
                for noise_layer in noise_layer_list:
                    for rand_seed in RAND_SEED_LIST:
                        for hidden_drop in hidden_drop_list:
                            for input_drop in input_drop_list:
                                for batchnorm in batchnorm_list:
                                    args_list.append([args.dataset, rand_seed, noise_layer,
                                        policy, nlayer, hdim, input_drop, hidden_drop,
                                        batchnorm])
def fully_train(hdim_list, nlayer,
                hidden_drop_list, input_drop_list, batchnorm_list):
    policy_list = ["IncomingNoisyMLP"]
    policy = "IncomingNoisyMLP"
    noise_layer = 0
    for hdim in hdim_list:
        for rand_seed in RAND_SEED_LIST:
            for hidden_drop in hidden_drop_list:
                for input_drop in input_drop_list:
                    for batchnorm in batchnorm_list:
                        args_list.append([args.dataset, rand_seed, noise_layer,
                            policy, nlayer, hdim, input_drop, hidden_drop,
                            batchnorm])
main()
