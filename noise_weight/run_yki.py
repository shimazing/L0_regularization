import os
import subprocess
import argparse
import time as time_
import datetime
from time import time
import random
import numpy as np
from multiprocessing import Pool

def run_exp(args):
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 train_mlp_yki.py --dataset {1} --model {2}"\
          " --rand_seed {3} --act_fn {4} --noise_layer {5} --policy {6}".format(
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
                    choices=["letter", "dna", "satimage", "ijcnn1", "cifar10","cifar100",
                             "abalone", "whitewine", "redwine"])
args = parser.parse_args()


def main(nlayer_list):
    i = 0
    n_process = 40
    nhid_list = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
    rand_seed_list = range(5)
    act_fn = "relu"
    noise_layer_list = [-1,0,1]
    policy_list = ["IncomingNoisyMLP", "AlternatingNoisyMLP"]
    args_list = []
    for nlayer in nlayer_list:
        model_name_list = gen_model_name_list(nhid_list, nlayer)
        for model_name in model_name_list:
            for rand_seed in rand_seed_list:
                for noise_layer in noise_layer_list:
                    if noise_layer == -1:
                        args_list.append([i % 4, args.dataset, model_name, rand_seed, act_fn, noise_layer, "IncomingNoisyMLP"])
                        i += 1
                    else:
                        for policy in policy_list:
                            args_list.append([i % 4, args.dataset, model_name, rand_seed, act_fn, noise_layer, policy])
                            i += 1
                    #args_list.append([0, model_name, rand_seed])
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process)
    pool.map(run_exp, args_list)


main([2,3,4])
#os.system("scp -r ckpt/ yki@143.248.57.168:/hdd01/MLILAB/research/noisynet/experiment/NoisyMLP/")
