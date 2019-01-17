import matplotlib.pyplot as plt
import numpy as np
import torch
import os

POLICY = "NoisyMLP"
SEED_MAX = 2
MARKER_SIZE = 30
def main():
    nhid_list = [100, 300, 500, 1000]
    for n_layer in [2,3,4]:
        #learning_result(nhid_list, n_layer)
        best_n_layer_mlp(nhid_list, n_layer)
        #epoch_n_layer_mlp(nhid_list, n_layer)

def get_graph_info(noise_layer):
    if noise_layer == -1:
        color = "black"
        marker = "o"
        label_name = "Fully training"
    elif noise_layer == 0:
        color = "C0"
        marker = "x"
        label_name = "Except 1st"
    elif noise_layer == 1:
        color = "C1"
        marker = "s"
        label_name = "Except 1st&2nd"
    else:
        raise ValueError
    return color, marker, label_name

def gen_model_name_list(nhid_list, n_layer):
    model_name_list = []
    for nhid in nhid_list:
        model_name_list.append("MLP"+"-{}".format(nhid)*n_layer)
    return model_name_list

def best_n_layer_mlp(nhid_list, n_layer):
    CKPT_DIR = "ckpt"
    #nhid_list = [100, 500, 1000]
    #n_layer = 3
    model_name_list = gen_model_name_list(nhid_list, n_layer)
    fix, ax = plt.subplots(1, 1, figsize=(10, 10))
    policy = "NoisyMLP"
    noise_layer_list = [-1, 0,1]
    for noise_layer in noise_layer_list:
        test_acc_list = []
        non_zero_list = []
        for model_name in model_name_list:
            acc_part = []
            for rand_seed in range(SEED_MAX):
                ckpt_name = "best_wine_{}_{}_{}_{}_{}.pth.tar".format(model_name, policy, noise_layer, "relu", rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except Exception  as  e:
                    print(e)
                    print(" *** Error opening {}".format(ckpt_name))
                    continue
                test_acc = ckpt["test_acc"]
                acc_part.append(test_acc)

            print("[(best){}_{}_{}] Test Acc: {:.2f}".format(policy, model_name,
                noise_layer, np.mean(acc_part) * 100))
            test_acc_list.append(np.mean(acc_part))
            non_zero_list.append(ckpt["non_zero_list"][-1])

        color, marker, label_name = get_graph_info(noise_layer)
        ax.scatter(np.log(non_zero_list), test_acc_list, alpha=0.8,
                color=color, marker=marker, s=MARKER_SIZE, label=label_name)

    ax.legend()
    ax.set_ylabel("Test Acc")
    ax.set_xlabel("log(num params)")
    ax.set_title("[{}] {} for WINE".format(POLICY, ", ".join(model_name_list)))
    plt.savefig("{}_{}_layer_MLP_WINE.png".format(POLICY, n_layer))
    plt.show()
    #plt.close()


main()
