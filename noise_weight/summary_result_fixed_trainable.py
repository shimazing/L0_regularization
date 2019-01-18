import os
import torch
import numpy as np
import matplotlib
import argparse
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

POLICY_list = ["IncomingNoisyMLP"]
SEED_MAX = 2
MARKER_SIZE = 80

def main():
    nhid_list = [100]
    n_layer_list = range(2, 10)
    #for n_layer in range(2, 10):
    learning_result(nhid_list, n_layer_list)#, n_layer)
    best_n_layer_mlp(nhid_list, n_layer_list)#, n_layer)
    epoch_n_layer_mlp(nhid_list, n_layer_list)#, n_layer)


def get_graph_info(policy, noise_layer):
    if policy == "AlternatingNoisyMLP":
        if noise_layer == 0:
            color = "C2"
            marker = "P"
            label_name = "Except Odd"
        elif noise_layer == 1:
            color = "C3"
            marker = "D"
            label_name = "Except Even"
        else:
            raise ValueError
    elif policy == "IncomingNoisyMLP":
        if noise_layer == -1:
            color = "black"
            marker = "o"
            label_name = "Fully training"
        elif noise_layer == 0:
            color = "C0"
            marker = "X"
            label_name = "Except 1st"
        elif noise_layer == 1:
            color = "C1"
            marker = "s"
            label_name = "Except 1st&2nd"
        else:
            raise ValueError
    else:
        raise ValueError
    return color, marker, label_name


def best_n_layer_mlp(nhid_list, n_layer_list):
    CKPT_DIR = "ckpt"
    # nhid_list = [100, 500, 1000]
    # n_layer = 3
    fix, ax = plt.subplots(1, 1, figsize=(20, 20))
    for n_layer in n_layer_list:
        noise_layer = n_layer - 3
        model_name_list = gen_model_name_list(nhid_list, n_layer)
        policy = "IncomingNoisyMLP"
        test_acc_list = []
        non_zero_list = []
        for model_name in model_name_list:
            acc_part = []
            for rand_seed in range(SEED_MAX):
                ckpt_name = "best_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset,model_name, policy, noise_layer, "relu", rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except:
                    print(" *** Error opening {}".format(ckpt_name))
                    continue
                test_acc = ckpt["test_auc"]
                acc_part.append(test_acc)
            print("[(best){}_{}_{}_{}] Test Auc: {:.2f}".format(args.dataset, policy, model_name,
                                                             noise_layer, np.mean(acc_part) * 100))

            test_acc_list.append(np.mean(acc_part))
            non_zero_list.append(ckpt["non_zero_list"][-1])

        #color, marker, label_name = get_graph_info(policy, noise_layer)
        color = np.random.rand(3)
        ax.scatter(np.log(non_zero_list), test_acc_list, alpha=0.5,
                label=model_name, c=color)
                   #color=color, marker=marker, s=MARKER_SIZE, label=label_name)

    ax.legend(loc="lower right")#, prop={'size': 30})
    ax.set_ylabel("Test Auc")
    ax.set_xlabel("log(num params)")
    ax.set_title("[{}] best".format(args.dataset))
    plt.show()
    plt.close()

def epoch_n_layer_mlp(nhid_list, n_layer_list, epoch="last"):
    CKPT_DIR = "ckpt"
    # nhid_list = [100, 500, 1000]
    # n_layer = 3
    fix, ax = plt.subplots(1, 1, figsize=(20, 20))
    for n_layer in n_layer_list:
        noise_layer = n_layer - 3
        model_name_list = gen_model_name_list(nhid_list, n_layer)
        policy = "IncomingNoisyMLP"
        test_acc_list = []
        non_zero_list = []
        for model_name in model_name_list:
            acc_part = []
            for rand_seed in range(SEED_MAX):
                if epoch == "last":
                    ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name, policy, noise_layer, "relu", rand_seed)
                else:
                    ckpt_name = "{}epoch_{}_{}_{}_{}_{}_{}.pth.tar".format(epoch, args.dataset, model_name, policy, noise_layer, "relu",
                                                                        rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except:
                    print(" *** Error opening {}".format(ckpt_name))
                    continue
                test_acc = ckpt["test_auc"]
                acc_part.append(test_acc)

            print("[({}epoch){}_{}_{}] Test Auc: {:.2f}".format(epoch, policy, model_name,
                                                                noise_layer, np.mean(acc_part) * 100))

            test_acc_list.append(np.mean(acc_part))
            non_zero_list.append(ckpt["non_zero_list"][-1])

        #color, marker, label_name = get_graph_info(policy, noise_layer)
        color = np.random.rand(3)
        ax.scatter(np.log(non_zero_list), test_acc_list, alpha=0.5,
                label=model_name, c=color)
                   #color=color, marker=marker, s=MARKER_SIZE, label=label_name)

    ax.legend(loc="lower right")#, prop={'size': 30})
    ax.set_ylabel("Test Auc")
    ax.set_xlabel("log(num params)")
    ax.set_title("[{}] {}-layered-MLPs at {} epoch".format(args.dataset, n_layer, epoch))
    plt.show()
    plt.savefig("{}_{}epoch_{}-layered-MLPs.png".format(args.dataset, epoch, n_layer))
    plt.show()
    plt.close()


def learning_result(nhid_list, n_layer_list):
    CKPT_DIR = "ckpt"
    # nhid_list = [100, 500, 1000]
    # n_layer = 3
    fix, ax = plt.subplots(1, 1, sharey="row",
                           figsize=(20, 20))
    for n_layer in n_layer_list:
        noise_layer = n_layer - 3
        model_name_list = gen_model_name_list(nhid_list, n_layer)
        policy = 'IncomingNoisyMLP'
        model_name = model_name_list[0]
        #for idx, model_name in enumerate(model_name_list):
        test_acc_list = []
        ax.set_title(model_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Auc")
        acc_part = []
        train_acc_list = []
        valid_acc_list = []
        for rand_seed in range(SEED_MAX):
            ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset,
                                                           model_name, policy, noise_layer, "relu", rand_seed)
            try:
                ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
            except:
                print(" *** Error opening {}".format(ckpt_name))
                continue
            test_acc = ckpt["test_auc"]
            acc_part.append(test_acc)
            train_acc_list.append(ckpt["train_auc_list"])
            valid_acc_list.append(ckpt["valid_auc_list"])

        #color, marker, label_name = get_graph_info(policy, noise_layer)
        label_name = model_name
        color = np.random.rand(3)
        ax.plot(np.mean(np.array(train_acc_list), axis=0),
                     alpha=0.5, c=color,
                     label=label_name + ", train")
        ax.plot(np.mean(np.array(valid_acc_list), axis=0),
                     alpha=0.5, c=color,
                     linestyle="--", label=label_name + ", valid")
        ax.legend(loc="lower right")
            # print("[{}_{}_{}] Test Acc: {:.2f}".format(policy, model_name,
            #    noise_layer, np.mean(acc_part) * 100))
            # test_acc_list.append(np.mean(acc_part))
        # ax.plot(np.arange(len(test_acc_list)), test_acc_list, label=policy)

        # ax.legend()
        # ax.set_ylabel("Test Acc.")
        # ax.set_xticklabels([""]+ model_name_list, rotation=60)
    # ax.set_title("[{}] Diff MLPs Accruacy for MNIST".format(POLICY))
    plt.show()
    plt.savefig("{}_Learning_results_{}-layered-MLP.png".format(args.dataset, n_layer))
    plt.close()


def gen_model_name_list(nhid_list, n_layer):
    model_name_list = []
    for nhid in nhid_list:
        model_name_list.append("MLP" + "-{}".format(nhid) * n_layer)
    return model_name_list


main()
