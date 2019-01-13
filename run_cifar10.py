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
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 train_lenet5.py --epochs 500 --dataset cifar10 \
    --verbose  --lambas {1} {1} {1} {1} {1} --sparsity {2} --rand_seed {3} \
    --name {4} --beta_ema {5}".format(
        args[0], args[1], args[2], args[3], args[4], args[5])
    print("[{}] {}".format(
        datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
        cmd
        ))
    time_.sleep(0.1)
    #os.system("julia MLF_Reg_Main.jl {0} {1} {2} {3} {4} {5} {6}".format(
    #    args[0], args[1], args[2], args[3], args[4], args[5], args[6]))
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    os.system(cmd)


parser = argparse.ArgumentParser()
args = parser.parse_args()
#####
beta_ema_list = [0.0]
sparsity_list = [0.0, 1.0]
models = ["L0LeNet5-6-16-120-84", "L0LeNet5-12-32-200-100", "L0LeNet5-24-64-400-200"]
lambda_list =  [1.4, 1.6]#, 0.4, 0.6, 0.8] #list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
# (01.06 19:41, server15~18) model==MLP-500-300, [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
# (01.06 21:41, server1, 11~14) model==MLP-1000-500, 2000-1000, [1e-6, 5e-6, 1e-5, 5e-5]+ list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001,0.01,0.001)) +list(np.arange(0.01, 0.1, 0.01))
# (01.07 12:41, server1, 10,11,15,16,17,18) model==MLP-300-100, [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
# (01.07 18:35, server1, 10,11,13,14,15,16,17,18) model==MLP-1000-500 and MLP-2000-1000, list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
i = 0
n_process = 40
args_list = []
for model in models:
    for s in sparsity_list:
        for l in lambda_list:
          for beta_ema in beta_ema_list:
            for rand_seed in range(4):
                args_list.append([i%4, l, s, rand_seed, model, beta_ema])
                i += 1
print("# Total training samples={}".format(len(args_list)))
#np.random.shuffle(args_list)
pool = Pool(processes=n_process)
pool.map(run_exp, args_list)

