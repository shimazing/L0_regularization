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
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 train_only_last_layer.py --model {1} \
     --rand_seed {2} --rank {3}".format(
        args[0], args[1], args[2], args[3])
    print("[{}] {}".format(
        datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
        cmd
        ))
    time_.sleep(0.1)
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    rank_list = [300, 150, 75]
    i = 0
    n_process = 40
    model_name_list = ["MLP-300-100"]
    #model_name_list = ["MLP-50-30", "MLP-100-50", "MLP-300-100", "MLP-500-300", "MLP-1000-500", "MLP-2000-1000"]
    rand_seed_list = range(10)
    args_list = []
    for model_name in model_name_list:
      for rank in rank_list:
        for rand_seed in rand_seed_list:
            args_list.append([i % 4, model_name, rand_seed, rank])
            i += 1
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process)
    pool.map(run_exp, args_list)
    #os.system("scp -r ckpt/ yki@143.248.57.168:/home/yki/Documents/compression/experiment/Rand_Noise_Expressive_Power/")

main()
