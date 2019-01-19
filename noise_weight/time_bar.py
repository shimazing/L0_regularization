import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import matplotlib
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

POLICY_list = ["NoisyVgg16"]
SEED_MAX = 4
MARKER_SIZE = 80

def main():
    hdim_list = [512, 1024, 2048, 4096]
    noisy_layer_list = range(11)
    best_n_layer_mlp(hdim_list, noisy_layer_list)

def best_n_layer_mlp(hdim_list, noise_layer_list):
    CKPT_DIR = "ckpt"
    policy = "NoisyVgg16"
    for nhid in hdim_list:
        fix, ax = plt.subplots(1, 1, figsize=(20, 20))
        forward_time_list = []
        backward_time_list = []
        optim_time_list = []
        for noise_layer in noise_layer_list:
            rand_seed = 13
            ckpt_name = "{}_{}_{}_{}_{}_{}_{}.pth.tar".format(
                    args.dataset, policy + "-{}".format(nhid),
                                                           'na', 'na',
                                                           noise_layer,
                                                           'relu',
                                                           rand_seed)
            try:
                ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
            except Exception as e:
                print(e)
                print(" *** Error opening {}".format(ckpt_name))
                continue
            forward_time_list.append(ckpt['forward_time'])
            backward_time_list.append(ckpt['backward_time'])
            optim_time_list.append(ckpt['optim_time'])
            #test_acc = ckpt["test_auc"]
            #color = np.random.rand(3)
        width = 0.35
        ax.bar(noise_layer_list, forward_time_list, width, label='forward')
        ax.bar(noise_layer_list, backward_time_list, width, label='backward',
                bottom=forward_time_list)
        ax.bar(noise_layer_list, optim_time_list, width, label='optim',
                bottom=[x+y for x, y in zip(forward_time_list,
                    backward_time_list)])

        ax.legend(loc="lower right")#, prop={'size': 30})
        ax.set_ylabel("Test Auc")
        ax.set_xlabel("log(num params)")
        ax.set_title("[{} fc_dim {}] best validation".format(args.dataset, nhid))
        plt.show()
        plt.close()
main()
