import os
import numpy as np
import torch
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import L0MLP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="MLP-300-100")
parser.add_argument("--policy", type=str, default="L0")
parser.add_argument("--n_param", type=float)
args = parser.parse_args()

CKPT_DIR = "runs"
POLICY = "L0_neuron"

def main():
    model_name_list = ["L0LeNet5-20-50-500", "L0LeNet5-40-75-1000", "L0LeNet-60-100-1500"]
    lambda_list =  [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2] #list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
    summary_best(model_name_list, lambda_list)
    summary_last_epoch(model_name_list, lambda_list)

def summary_last_epoch(model_name_list, coef_list):
    # coef_list = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    # model_name_list = ["MLP-300-100", "MLP-500-300", "MLP-1000-500", "MLP-2000-1000"]
    rand_seed_list = range(8)
    sparsity_list = [0.0, 1.0]
    test_acc_dict = {}
    test_acc_avg_dict = {}
    non_zeros_dict = {}
    non_zeros_avg_dict = {}

    for model_name in model_name_list:
        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            test_acc_list = []
            test_acc_avg_list = []
            non_zeros_list = []
            non_zeros_avg_list = []
            for coef in coef_list:
                acc_part = []
                non_zeros_part = []
                for rand_seed in rand_seed_list:
                    ckpt_name = ("{0}_{1}_policy_{2}_{3:.2f}_noise_{4:.3f}" +
                            "_{5:.2e}" * 4  + \
                            "_{6}_{7:.2f}.pth.tar").format(
                        model_name, args.policy, rand_seed, sparsity, 0.999,
                        coef, False, 0.67
                    )
                    if os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name,'checkpoint.pth.tar'))
                    else:
                        print(" *** Missing {}".format(ckpt_name))
                        continue
                    features = ckpt["pruned_model"]
                    non_zero = np.sum(features)
                    test_acc = ckpt["test_acc"]

                    acc_part.append(test_acc)
                    non_zeros_part.append(non_zero)

                if len(acc_part) > 0:
                    test_acc_avg_list.append(np.mean(acc_part))
                    non_zeros_avg_list.append(np.mean(non_zeros_part))
                    test_acc_list += acc_part
                    non_zeros_list += non_zeros_part

            test_acc_dict[label_name] = test_acc_list
            non_zeros_dict[label_name] = non_zeros_list
            test_acc_avg_dict[label_name] = test_acc_avg_list
            non_zeros_avg_dict[label_name] = non_zeros_avg_list

    for i in range(len(model_name_list)):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        for model_name in model_name_list[:i + 1]:

            for sparsity in sparsity_list:
                if sparsity == 0.0:
                    label_name = "{} w noise".format(model_name)
                else:
                    label_name = "{} w/o noise".format(model_name)
                ax[0].scatter(np.log(non_zeros_avg_dict[label_name]), test_acc_avg_dict[label_name],
                              alpha=0.5, label=label_name)
                ax[1].scatter(np.log(non_zeros_dict[label_name]), test_acc_dict[label_name],
                              alpha=0.5, label=label_name)
                idx = np.argsort(0.0 - np.array(test_acc_dict[label_name]))
                for j in idx[:5]:
                    print("[{}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(label_name,
                                                                                  100 - 100 * test_acc_dict[label_name][
                                                                                      j],
                                                                                  np.log(
                                                                                      non_zeros_dict[label_name][j])))

        for ax_ in [ax[0], ax[1]]:
            ax_.legend(loc="lower right")
            #ax_.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
            #ax_.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
            ax_.axhline(y=0.991, color="red", alpha=0.3, linestyle="--")
            ax_.set_ylabel("Test Acc.")
            ax_.set_xlabel("log(Non-zeros)")
        ax[0].set_title("{} policy, (avg.) Results".format(POLICY))
        ax[1].set_title("{} policy, Results".format(POLICY))
        plt.show()#savefig("last_epoch_{}_{}.png".format(POLICY, "_".join(model_name_list[:i + 1])))
        for ax_ in [ax[0], ax[1]]:
            ax_.set_yticks([0.97, 0.982, 0.984, 0.986])
            ax_.set_ylim(0.97, 0.99)
        plt.show()#savefig("last_epoch_{}_{}_acc_crop.png".format(POLICY, "_".join(model_name_list[:i + 1])))
        #plt.close()

        if i == 0:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            ax[0].scatter(np.log(non_zeros_avg_dict[label_name]), test_acc_avg_dict[label_name],
                          alpha=0.5, label=label_name)
            ax[1].scatter(np.log(non_zeros_dict[label_name]), test_acc_dict[label_name],
                          alpha=0.5, label=label_name)
        for ax_ in [ax[0], ax[1]]:
            ax_.legend(loc="lower right")
            #ax_.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
            #ax_.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
            ax_.axhline(y=0.991, color="red", alpha=0.3, linestyle="--")
            ax_.set_ylabel("Test Acc.")
            ax_.set_xlabel("log(Non-zeros)")
        ax[0].set_title("{} policy, (avg.) Results".format(POLICY))
        ax[1].set_title("{} policy, Results".format(POLICY))
        plt.show()#savefig("last_epoch_{}_{}.png".format(POLICY, model_name))
        for ax_ in [ax[0], ax[1]]:
            ax_.set_yticks([0.97, 0.982, 0.984, 0.986])
            ax_.set_ylim(0.97, 0.99)
        plt.show()#savefig("last_epoch_{}_{}_acc_crop.png".format(POLICY, model_name))
        #plt.close()


def summary_best(model_name_list, coef_list):
    #coef_list = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    # model_name_list = ["MLP-500-300", "MLP-300-100", "MLP-1000-500"]
    rand_seed_list = range(8)
    sparsity_list = [0.0, 1.0]
    test_acc_dict = {}
    test_acc_avg_dict = {}
    non_zeros_dict = {}
    non_zeros_avg_dict = {}

    for model_name in model_name_list:
        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            test_acc_list = []
            test_acc_avg_list = []
            non_zeros_list = []
            non_zeros_avg_list = []
            for coef in coef_list:
                acc_part = []
                non_zeros_part = []
                for rand_seed in rand_seed_list:
                    ckpt_name = ("{0}_{1}_policy_{2}_{3:.2f}_noise_{4:.3f}" +
                            "_{5:.2e}" * 4  + \
                            "_{6}_{7:.2f}.pth.tar").format(
                        model_name, args.policy, rand_seed, sparsity, 0.999,
                        coef, False, 0.67
                    )
                    if os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        ckpt = torch.load(os.path.join(CKPT_DIR,
                            ckpt_name,'model_best.pth.tar'))
                    else:
                        print(" *** Missing {}".format(ckpt_name))
                        continue
                    features = ckpt["pruned_model"]
                    non_zero = np.sum(features)
                    test_acc = ckpt["test_acc"]

                    acc_part.append(test_acc)
                    non_zeros_part.append(non_zero)

                if len(acc_part) > 0:
                    test_acc_avg_list.append(np.mean(acc_part))
                    non_zeros_avg_list.append(np.mean(non_zeros_part))
                    test_acc_list += acc_part
                    non_zeros_list += non_zeros_part

            test_acc_dict[label_name] = test_acc_list
            non_zeros_dict[label_name] = non_zeros_list
            test_acc_avg_dict[label_name] = test_acc_avg_list
            non_zeros_avg_dict[label_name] = non_zeros_avg_list

    for i in range(len(model_name_list)):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        for model_name in model_name_list[:i+1]:
            for sparsity in sparsity_list:
                if sparsity == 0.0:
                    label_name = "{} w noise".format(model_name)
                else:
                    label_name = "{} w/o noise".format(model_name)
                ax[0].scatter(np.log(non_zeros_avg_dict[label_name]), test_acc_avg_dict[label_name],
                              alpha=0.5, label=label_name)
                ax[1].scatter(np.log(non_zeros_dict[label_name]), test_acc_dict[label_name],
                              alpha=0.5, label=label_name)
                idx = np.argsort(0.0 - np.array(test_acc_dict[label_name]))
                for j in idx[:5]:
                    print("[{}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(label_name,
                                                                                  100-100*test_acc_dict[label_name][j],
                                                                                  np.log(non_zeros_dict[label_name][j])))

        for ax_ in [ax[0], ax[1]]:
            ax_.legend(loc="lower right")
            #ax_.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
            #ax_.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
            ax_.axhline(y=0.991, color="red", alpha=0.3, linestyle="--")
            ax_.set_ylabel("Test Acc.")
            ax_.set_xlabel("log(Non-zeros)")
        ax[0].set_title("{} policy, (avg.) Results".format(POLICY))
        ax[1].set_title("{} policy, Results".format(POLICY))
        plt.show()#("best_{}_{}.png".format(POLICY, "_".join(model_name_list[:i+1])))
        for ax_ in [ax[0], ax[1]]:
            ax_.set_yticks([0.97, 0.982, 0.984, 0.986])
            ax_.set_ylim(0.97, 0.99)
        plt.show()#savefig("best_{}_{}_acc_crop.png".format(POLICY, "_".join(model_name_list[:i+1])))
        #plt.close()

        if i == 0:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            ax[0].scatter(np.log(non_zeros_avg_dict[label_name]), test_acc_avg_dict[label_name],
                          alpha=0.5, label=label_name)
            ax[1].scatter(np.log(non_zeros_dict[label_name]), test_acc_dict[label_name],
                          alpha=0.5, label=label_name)
        for ax_ in [ax[0], ax[1]]:
            ax_.legend(loc="lower right")
            ax_.axhline(y=0.991, color="red", alpha=0.3, linestyle="--")
            ax_.set_ylabel("Test Acc.")
            ax_.set_xlabel("log(Non-zeros)")
        ax[0].set_title("{} policy, (avg.) Results".format(POLICY))
        ax[1].set_title("{} policy, Results".format(POLICY))
        plt.show()#savefig("best_{}_{}.png".format(POLICY, model_name))
        for ax_ in [ax[0], ax[1]]:
            ax_.set_yticks([0.97, 0.982, 0.984, 0.986])
            ax_.set_ylim(0.97, 0.99)
        plt.show()#savefig("best_{}_{}_acc_crop.png".format(POLICY, model_name))
        #plt.close()

main()
