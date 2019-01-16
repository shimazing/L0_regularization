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
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 run.py --experiment {1} --approach {2} --parameter {3} --seed {4}".format(
        args[0], args[1], args[2], args[3], args[4])
    print("[{}] {}".format(
        datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
        cmd
        ))
    time_.sleep(0.1)
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    os.system(cmd)

parser = argparse.ArgumentParser()
#parser.add_argument("--parameter", type=str, required=True)
#parser.add_argument("--approach", type=str, choices=["hat","noisy-hat"])
parser.add_argument("--experiment", type=str, default="cifar")
args = parser.parse_args()

def main():
    i = 0
    n_process = 4
    rand_seed_list = range(4)
    args_list = []
    for approach in ["noisy-hat"]:
     for parameter in ["0.2,400", "0.4,400", "0.8,400","1.6,400", "3.2,400"]:
      for rand_seed in rand_seed_list:
        args_list.append([i % 4, args.experiment, approach, parameter, rand_seed])
        i += 1
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process)
    pool.map(run_exp, args_list)
    #os.system("scp -r ckpt/ yki@143.248.57.168:/home/yki/Documents/continual-learning/experiment/NoisyHAT/")
    #os.system("scp -r res/ yki@143.248.57.168:/home/yki/Documents/continual-learning/experiment/NoisyHAT/")

main()
