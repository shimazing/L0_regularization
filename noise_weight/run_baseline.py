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
    cmd = "CUDA_VISIBLE_DEVICES={0} python3 baselines.py --dataset {5} \
            --model {1} --rand_seed {2} --C {3} --kernel {4} ".format(
        args[0], args[1], args[2], args[3], args[4], args[5])
    print("[{}] {}".format(
        datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
        cmd
        ))
    time_.sleep(0.1)
    #subprocess.run(cmd.split(" "), check=True, stdout=subprocess.PIPE)
    os.system(cmd)

def main():
    i = 0
    n_process = 20
    model_name_list = ['svm', 'logisticRegression']
    dataset_list = ['abalone'] #['wine', 'redwine', 'abalone', 'dna', "letter", "satimage",
    #        "ijcnn1",]
    C_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    kernel_list = ['rbf', 'linear']
    #return
    rand_seed_list = range(5)
    act_fn = "relu"
    noise_layer_list = [-1,0,1]
    args_list = []
    for model_name in model_name_list:
      for dataset in dataset_list:
        for rand_seed in rand_seed_list:
          for kernel in kernel_list:
            for C in C_list:
                if model_name == 'logisticRegression' and kernel == 'rbf':
                    continue
                args_list.append([i % 4, model_name, rand_seed, C, kernel,
                    dataset])
                i += 1
                #args_list.append([0, model_name, rand_seed])
    print("# Total training samples={}".format(len(args_list)))
    np.random.shuffle(args_list)
    pool = Pool(processes=n_process)
    pool.map(run_exp, args_list)

main()
#os.system("scp -r ckpt/ mazing@server03.mli.kr:/home/mazing/L0_regularization/noise_weight/ckpt/")
