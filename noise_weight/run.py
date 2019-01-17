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
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 train_mlp.py --dataset wine --model {1} --rand_seed {2} --act_fn {3} --noise_layer {4}".format(
        args[0], args[1], args[2], args[3], args[4])
    print("[{}] {}".format(
        datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
        cmd
        ))
    time_.sleep(0.1)
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    #os.system(cmd)

def gen_model_name_list(nhid_list, n_layer):
    model_name_list = []
    for nhid in nhid_list:
        model_name_list.append("MLP"+"-{}".format(nhid)*n_layer)
    return model_name_list

def main(nlayer):
    # parser = argparse.ArgumentParser()
    #parser.add_argument("--nlayer", type=int, required=True)
    #args = parser.parse_args()

    i = 0
    n_process = 20
    #model_name_list = ["MLP-50-30", "MLP-100-50", "MLP-300-100", "MLP-500-300", "MLP-1000-500", "MLP-2000-1000"]
    #model_name_list = ["MLP-3000-2000", "MLP-4000-3000", "MLP-5000-4000", "MLP-6000-5000", "MLP-7000-6000", "MLP-8000-7000", "MLP-9000-8000", "MLP-10000-9000", "MLP-10000-10000"]
    #model_name_list = ["MLP-100-100-100", "MLP-200-200-200", "MLP-500-500-500", "MLP-1000-1000-1000", "MLP-2000", "MLP-5000","MLP-10000", "MLP-20000"]
    nhid_list = [100, 300, 500, 1000]
    model_name_list = gen_model_name_list(nhid_list, nlayer)
    print(model_name_list)
    #return
    rand_seed_list = range(2)
    act_fn = "relu"
    noise_layer_list = [-1,0,1]
    args_list = []
    for model_name in model_name_list:
        for rand_seed in rand_seed_list:
            for noise_layer in noise_layer_list:
                args_list.append([i % 4, model_name, rand_seed, act_fn, noise_layer])
                i += 1
                #args_list.append([0, model_name, rand_seed])
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process)
    pool.map(run_exp, args_list)

main(2)
main(3)
main(4)
os.system("scp -r ckpt/ mazing@server03.mli.kr:/home/mazing/L0_regularization/noise_weight/ckpt/")
