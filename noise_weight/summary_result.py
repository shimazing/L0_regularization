import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY = "NoisyMLP"
SEED_MAX = 2
def main():
    nhid_list = [100, 300, 500, 1000]
    for n_layer in [2,3, 4]:
        #learning_result(nhid_list, n_layer)
        #best_n_layer_mlp(nhid_list, n_layer)
        epoch_n_layer_mlp(nhid_list, n_layer)

def best_n_layer_mlp(nhid_list, n_layer):
    CKPT_DIR = "ckpt"
    #nhid_list = [100, 500, 1000]
    #n_layer = 3
    model_name_list = gen_model_name_list(nhid_list, n_layer)
    fix, ax = plt.subplots(1, 1, figsize=(10, 10))
    policy = "NoisyMLP"
    noise_layer_list = [-1, 0,1]
    for model_name in model_name_list:
        test_acc_list = []
        for noise_layer in noise_layer_list:
            acc_part = []
            for rand_seed in range(SEED_MAX):
                ckpt_name = "best_wine_{}_{}_{}_{}_{}.pth.tar".format(model_name, policy, noise_layer, "relu", rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except:
                    #print(" *** Error opening {}".format(ckpt_name))
                    continue
                test_acc = ckpt["test_acc"]
                acc_part.append(test_acc)

            print("[(best){}_{}_{}] Test Acc: {:.2f}".format(policy, model_name,
                noise_layer, np.mean(acc_part) * 100))
            #test_acc_list.append(np.mean(acc_part))
        #ax.plot(np.arange(len(test_acc_list)), test_acc_list, label=policy)

    #ax.legend()
    #ax.set_ylabel("Test Acc.")
    #ax.set_xticklabels([""]+ model_name_list, rotation=60)
    #ax.set_title("[{}] Diff MLPs Accruacy for MNIST".format(POLICY))
    #plt.savefig("{}_two_layer_MLP_MNIST.png".format(POLICY))
    plt.close()

def epoch_n_layer_mlp(nhid_list, n_layer,epoch="last"):
    SEED_MAX = 2
    CKPT_DIR = "ckpt"
    #nhid_list = [100, 500, 1000]
    #n_layer = 3
    model_name_list = gen_model_name_list(nhid_list, n_layer)
    fix, ax = plt.subplots(1, 1, figsize=(10, 10))
    policy = "NoisyMLP"
    noise_layer_list = [-1,0,1]
    for model_name in model_name_list:
        test_acc_list = []
        for noise_layer in noise_layer_list:
            acc_part = []
            for rand_seed in range(SEED_MAX):
                if epoch=="last":
                    ckpt_name = "wine_{}_{}_{}_{}_{}.pth.tar".format(model_name, policy, noise_layer, "relu", rand_seed)
                else:
                    ckpt_name = "{}epoch_wine_{}_{}_{}_{}_{}.pth.tar".format(epoch, model_name, policy, noise_layer, "relu", rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except:
                    print(" *** Error opening {}".format(ckpt_name))
                    continue
                test_acc = ckpt["test_acc"]
                acc_part.append(test_acc)

            print("[({}epoch){}_{}_{}] Test Acc: {:.2f}".format(epoch, policy, model_name,
                noise_layer, np.mean(acc_part) * 100))
            #test_acc_list.append(np.mean(acc_part))
        #ax.plot(np.arange(len(test_acc_list)), test_acc_list, label=policy)

    #ax.legend()
    #ax.set_ylabel("Test Acc.")
    #ax.set_xticklabels([""]+ model_name_list, rotation=60)
    #ax.set_title("[{}] Diff MLPs Accruacy for MNIST".format(POLICY))
    #plt.savefig("{}_two_layer_MLP_MNIST.png".format(POLICY))
    plt.close()


def learning_result(nhid_list, n_layer):

    CKPT_DIR = "ckpt"
    #nhid_list = [100, 500, 1000]
    #n_layer = 3
    model_name_list = gen_model_name_list(nhid_list, n_layer)
    fix, ax = plt.subplots(1, len(model_name_list), figsize=(5*len(model_name_list), 5))
    policy = "NoisyMLP"
    noise_layer_list = [-1,0,1]
    for idx, model_name in enumerate(model_name_list):
        test_acc_list = []
        ax[idx].set_title(model_name)
        ax[idx].set_xlabel("Epoch")
        ax[idx].set_ylabel("Acc")

        for noise_layer in noise_layer_list:
            acc_part = []
            train_acc_list = []
            valid_acc_list = []
            for rand_seed in range(SEED_MAX):
                ckpt_name = "wine_{}_{}_{}_{}_{}.pth.tar".format(model_name, policy, noise_layer, "relu", rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except:
                    print(" *** Error opening {}".format(ckpt_name))
                    continue
                test_acc = ckpt["test_acc"]
                acc_part.append(test_acc)
                train_acc_list.append(ckpt["train_acc_list"])
                valid_acc_list.append(ckpt["valid_acc_list"])

            if noise_layer == -1:
                color = "black"
                label_name = "Fully training"
            elif noise_layer == 0:
                #color = "C0"
                color = "r"
                label_name = "Except 1st"
            elif noise_layer == 1:
                #color = "C1"
                color = "b"
                label_name = "Except 1st&2nd"
            else:
                raise ValueError

            ax[idx].plot(np.mean(np.array(train_acc_list),axis=0),
                    alpha=0.5, c=color, label=label_name+", train")
            ax[idx].plot(np.mean(np.array(valid_acc_list),axis=0),
                    alpha=0.5, c=color, linestyle="--", label=label_name+", valid")
            ax[idx].legend()
            #print("[{}_{}_{}] Test Acc: {:.2f}".format(policy, model_name,
            #    noise_layer, np.mean(acc_part) * 100))
            #test_acc_list.append(np.mean(acc_part))
        #ax.plot(np.arange(len(test_acc_list)), test_acc_list, label=policy)

    #ax.legend()
    #ax.set_ylabel("Test Acc.")
    #ax.set_xticklabels([""]+ model_name_list, rotation=60)
    #ax.set_title("[{}] Diff MLPs Accruacy for MNIST".format(POLICY))
    plt.savefig("Learning_results_{}-layered-MLP.png".format(n_layer))
    plt.close()

def gen_model_name_list(nhid_list, n_layer):
    model_name_list = []
    for nhid in nhid_list:
        model_name_list.append("MLP"+"-{}".format(nhid)*n_layer)
    return model_name_list

main()
