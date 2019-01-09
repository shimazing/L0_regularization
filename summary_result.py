import os
import numpy as np

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import L0MLP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="MLP-300-100")
parser.add_argument("--policy", type=str, default="L0")
parser.add_argument("--init", type=str, default="rand")
parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--n_param", type=float)
args = parser.parse_args()

def summary09():
    sparsity_list = [0.0, 1.0]
    model_list = ["L0LeNet5-20-50-500", "L0LeNet5-40-75-1000", "L0LeNet-60-100-1500"]
    lambda_list =  [0.001, 0.01, 0.1, 0.2, 0.4] #list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
    #np.random.seed(1234)
    #sparsity_list = [0.0, 1.0]
    #n_param_list = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0,
    #n_param_list = list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
    # To check (after 01.06.19:41)
    #n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
    rand_seed_list = range(8)
    CKPT_DIR = "runs" # TODO check
    for model_name in model_list:
        n_param_list = lambda_list

        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            test_acc_list = []
            ratio_list = []
            epoch = []
            for n_param in n_param_list:
                ratio_part = []
                acc_part = []
                for rand_seed in rand_seed_list:
                    # "LeNet-300-100_L0_policy_0_0.0_noise_0.999_{0:.2e}_{0:.2e}_{0:.2e}_False_0.67.pth.tar".format(p)
                    ckpt_name = ("{0}_{1}_policy_{2}_{3:.2f}_noise_{4:.3f}" + "_{5:.2e}" * len(args.lambas) + \
                            "_{6}_{7:.2f}.pth.tar").format(
                        model_name, args.policy, rand_seed, sparsity, 0.999,
                        n_param, False, 0.67
                    )
                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        print(" *** Misiing: ", ckpt_name)
                        continue
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name,'model_best.pth.tar'))
                    features = ckpt["pruned_model"]
                    non_zero = np.sum(features)
                    ratio_part.append(np.log(non_zero))
                    test_acc = ckpt["test_acc"]
                    acc_part.append(test_acc)
                #ratio_list.append(np.mean(ratio_part))
                #test_acc_list.append(np.mean(acc_part))
                ratio_list += ratio_part
                test_acc_list += acc_part

            #print("sparsity={:.1f}, epochs={}".format(sparsity, np.mean(epoch)))
            plt.scatter(ratio_list, test_acc_list, alpha=0.5, label=label_name)
            idx = np.argsort(0.0 - np.array(test_acc_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))
            idx = np.argsort(np.array(ratio_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))

    plt.legend(loc="lower right")
    #plt.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
    #plt.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
    #plt.axhline(y=0.986, color="red", alpha=0.3, linestyle="--")
    #plt.axvline(x=10.19, color="black", alpha=0.3, linestyle="--")
    #plt.axvline(x=11.15, color="black", alpha=0.3, linestyle="--")
    #plt.xticks([0, 3.59, 10, 20, 26.02, 30, 40, 50], rotation="vertical")
    #plt.title("L0 policy, Results (avg. 10 trials)")
    plt.ylabel("Test Acc.")
    plt.xlabel("log(Non-zero)")
    plt.savefig("l0_policy_diff_models_avg.png")
    plt.yticks([0.97, 0.982, 0.984, 0.986])
    plt.ylim(0.97, 0.99)
    plt.savefig("l0_policy_diff_models_avg_acc_crop.png")
    plt.show()
    plt.close()

def summary08():
    np.random.seed(1234)
    sparsity_list = [0.0, 1.0]
    #n_param_list = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0,
    #n_param_list = list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
    # To check (after 01.06.19:41)
    #n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
    rand_seed_list = range(10)
    CKPT_DIR = "ckpt"
    model_list = ["MLP-300-100", "MLP-500-300", "MLP-1000-500", "MLP-2000-1000"]
    for model_name in model_list:
        if model_name == "MLP-300-100" or model_name == "MLP-500-300":
            n_param_list = [1e-6, 5e-6, 1e-5, 5e-5]+ list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001,0.01,0.001)) + list(np.arange(0.01, 0.1, 0.01)) +  list(np.arange(0.1,0.2,0.01)) + list(np.arange(0.3,  0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2))
        elif model_name == "MLP-300-100":
            n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))  #list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
        elif model_name == "MLP-500-300":
            n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))  #list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
        elif model_name == "MLP-1000-500" or model_name == "MLP-2000-1000":
            n_param_list = [1e-6, 5e-6, 1e-5, 5e-5] + list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) # + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
        else:
            raise ValueError

        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            test_acc_list = []
            ratio_list = []
            epoch = []
            for n_param in n_param_list:
                ratio_part = []
                acc_part = []
                for rand_seed in rand_seed_list:
                    # "LeNet-300-100_L0_policy_0_0.0_noise_0.999_{0:.2e}_{0:.2e}_{0:.2e}_False_0.67.pth.tar".format(p)
                    ckpt_name = "{0}_{1}_policy_{2}_{3:.2f}_noise_0.999_{4:.2e}_{4:.2e}_{4:.2e}_False_0.67.pth.tar".format(
                        model_name, args.policy, rand_seed, sparsity, n_param)
                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        print(" *** Misiing: ", ckpt_name)
                        continue
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    #if sparsity == 0.0:
                    #    print(ckpt_name)
                    token = model_name.split("-")
                    layer1_dim = int(token[1])
                    layer2_dim = int(token[2])
                    model = L0MLP(784,10, layer_dims=(layer1_dim, layer2_dim))
                    model.cuda()
                    model.load_state_dict(ckpt["model_state_dict"])
                    features = model.compute_params()

                    total = 0
                    non_zero = 0
                    for i in range(len(features) - 1):
                        non_zero += features[i] * features[i + 1]
                    non_zero += features[-1] * 10
                    epoch.append(ckpt["epoch"])
                    #for k in ckpt["model_state_dict"].keys():
                    #    if "noise" in k or "qz" in k:
                    #        continue
                    #    total += np.prod(ckpt["model_state_dict"][k].size())

                    #ratio_part.append(non_zero_ratio)
                    ratio_list.append(np.log(non_zero))
                    test_acc = ckpt["test_acc"]
                    #acc_part.append(test_acc)
                    test_acc_list.append(test_acc)
                #ratio_list.append(np.mean(ratio_part))
                #test_acc_list.append(np.mean(acc_part))
            #print("sparsity={:.1f}, epochs={}".format(sparsity, np.mean(epoch)))
            plt.scatter(ratio_list, test_acc_list, alpha=0.2, label=label_name)
            idx = np.argsort(0.0 - np.array(test_acc_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))
            idx = np.argsort(np.array(ratio_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))

    plt.legend(loc="lower right")
    plt.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
    plt.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
    plt.axhline(y=0.986, color="red", alpha=0.3, linestyle="--")
    plt.axvline(x=10.19, color="black", alpha=0.3, linestyle="--")
    plt.axvline(x=11.15, color="black", alpha=0.3, linestyle="--")
    #plt.xticks([0, 3.59, 10, 20, 26.02, 30, 40, 50], rotation="vertical")
    plt.title("L0 policy, Results")
    plt.ylabel("Test Acc.")
    plt.xlabel("log(Non-zero)")
    plt.savefig("l0_policy_diff_models.png")
    plt.yticks([0.97, 0.982, 0.984, 0.986])
    plt.ylim(0.97, 0.99)
    plt.savefig("l0_policy_diff_models_acc_crop.png")
    plt.close()

def summary07():
    np.random.seed(1234)
    sparsity_list = [0.0, 1.0]
    rand_seed_list = range(10)
    CKPT_DIR = "ckpt"
    model_list = ["MLP-300-100", "MLP-500-300"] #, "MLP-1000-500", "MLP-2000-1000"]
    for model_name in model_list:

        if model_name == "MLP-300-100" or model_name == "MLP-500-300":
            n_param_list = [1e-6, 5e-6, 1e-5, 5e-5]+ list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001,0.01,0.001)) + list(np.arange(0.01, 0.1, 0.01)) +  list(np.arange(0.1,0.2,0.01)) + list(np.arange(0.3,  0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2))
        elif model_name == "MLP-300-100":
            n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))  #list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
        elif model_name == "MLP-500-300":
            n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))  #list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
        elif model_name == "MLP-1000-500" or model_name == "MLP-2000-1000":
            n_param_list = [1e-6, 5e-6, 1e-5, 5e-5] + list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) # + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
        else:
            raise ValueError

        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            test_acc_list = []
            ratio_list = []
            epoch = []
            for n_param in n_param_list:
                ratio_part = []
                acc_part = []
                for rand_seed in rand_seed_list:
                    # "LeNet-300-100_L0_policy_0_0.0_noise_0.999_{0:.2e}_{0:.2e}_{0:.2e}_False_0.67.pth.tar".format(p)
                    ckpt_name = "{0}_{1}_policy_{2}_{3:.2f}_noise_0.999_{4:.2e}_{4:.2e}_{4:.2e}_False_0.67.pth.tar".format(
                        model_name, args.policy, rand_seed, sparsity, n_param)
                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        print(" *** Misiing: ", ckpt_name)
                        continue
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    #if sparsity == 0.0:
                    #    print(ckpt_name)
                    token = model_name.split("-")
                    layer1_dim = int(token[1])
                    layer2_dim = int(token[2])
                    model = L0MLP(784,10, layer_dims=(layer1_dim, layer2_dim))
                    model.cuda()
                    model.load_state_dict(ckpt["model_state_dict"])
                    features = model.compute_params()

                    total = 0
                    non_zero = 0
                    for i in range(len(features) - 1):
                        non_zero += features[i] * features[i + 1]
                    non_zero += features[-1] * 10
                    epoch.append(ckpt["epoch"])
                    #for k in ckpt["model_state_dict"].keys():
                    #    if "noise" in k or "qz" in k:
                    #        continue
                    #    total += np.prod(ckpt["model_state_dict"][k].size())

                    ratio_part.append(np.log(non_zero))
                    #ratio_list.append(np.log(non_zero))
                    test_acc = ckpt["test_acc"]
                    acc_part.append(test_acc)
                    #test_acc_list.append(test_acc)
                ratio_list.append(np.mean(ratio_part))
                test_acc_list.append(np.mean(acc_part))
            #print("sparsity={:.1f}, epochs={}".format(sparsity, np.mean(epoch)))
            plt.scatter(ratio_list, test_acc_list, alpha=0.5, label=label_name)
            idx = np.argsort(0.0 - np.array(test_acc_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))
            idx = np.argsort(np.array(ratio_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))

    plt.legend(loc="lower right")
    plt.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
    plt.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
    plt.axhline(y=0.986, color="red", alpha=0.3, linestyle="--")
    plt.axvline(x=10.19, color="black", alpha=0.3, linestyle="--")
    plt.axvline(x=11.15, color="black", alpha=0.3, linestyle="--")
    #plt.xticks([0, 3.59, 10, 20, 26.02, 30, 40, 50], rotation="vertical")
    plt.title("L0 policy, Results (avg. 10 trials)")
    plt.ylabel("Test Acc.")
    plt.xlabel("log(Non-zero)")
    plt.savefig("l0_policy_diff_two_models_avg.png")
    plt.yticks([0.97, 0.982, 0.984, 0.986])
    plt.ylim(0.97, 0.99)
    plt.savefig("l0_policy_diff_two_models_avg_acc_crop.png")
    plt.close()

def summary06():
    np.random.seed(1234)
    sparsity_list = [0.0, 1.0]
    #n_param_list = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0,
    #n_param_list = list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))
    # To check (after 01.06.19:41)
    #n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
    rand_seed_list = range(10)
    CKPT_DIR = "ckpt"
    model_list = ["MLP-300-100", "MLP-500-300"] #, "MLP-1000-500", "MLP-2000-1000"]
    for model_name in model_list:
        if model_name == "MLP-300-100" or model_name == "MLP-500-300":
            n_param_list = [1e-6, 5e-6, 1e-5, 5e-5]+ list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001,0.01,0.001)) + list(np.arange(0.01, 0.1, 0.01)) +  list(np.arange(0.1,0.2,0.01)) + list(np.arange(0.3,  0.4, 0.02)) + list(np.arange(0.4, 1.1, 0.2))
        elif model_name == "MLP-300-100":
            n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))  #list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
        elif model_name == "MLP-500-300":
            n_param_list = [1e-5, 5e-5, 1e-4, 5e-4] + list(np.arange(0.001,0.01,0.002)) +list(np.arange(0.01, 0.1, 0.02)) + list(np.arange(0.1, 0.15, 0.01)) + list(np.arange(0.15, 0.2, 0.01))+ list(np.arange(0.3, 0.4, 0.02))  #list(np.arange(0.4, 1.1, 0.2)) + [1.5, 2.0, 2.5, 3.0, 5.0]
        elif model_name == "MLP-1000-500" or model_name == "MLP-2000-1000":
            n_param_list = [1e-6, 5e-6, 1e-5, 5e-5] + list(np.arange(0.0001, 0.001, 0.0002)) + list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) # + list(np.arange(0.1, 0.2, 0.01)) + list(np.arange(0.3, 0.4, 0.02))
        else:
            raise ValueError

        for sparsity in sparsity_list:
            if sparsity == 0.0:
                label_name = "{} w noise".format(model_name)
            else:
                label_name = "{} w/o noise".format(model_name)
            test_acc_list = []
            ratio_list = []
            epoch = []
            for n_param in n_param_list:
                ratio_part = []
                acc_part = []
                for rand_seed in rand_seed_list:
                    # "LeNet-300-100_L0_policy_0_0.0_noise_0.999_{0:.2e}_{0:.2e}_{0:.2e}_False_0.67.pth.tar".format(p)
                    ckpt_name = "{0}_{1}_policy_{2}_{3:.2f}_noise_0.999_{4:.2e}_{4:.2e}_{4:.2e}_False_0.67.pth.tar".format(
                        model_name, args.policy, rand_seed, sparsity, n_param)
                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        print(" *** Misiing: ", ckpt_name)
                        continue
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    #if sparsity == 0.0:
                    #    print(ckpt_name)
                    token = model_name.split("-")
                    layer1_dim = int(token[1])
                    layer2_dim = int(token[2])
                    model = L0MLP(784,10, layer_dims=(layer1_dim, layer2_dim))
                    model.cuda()
                    model.load_state_dict(ckpt["model_state_dict"])
                    features = model.compute_params()

                    total = 0
                    non_zero = 0
                    for i in range(len(features) - 1):
                        non_zero += features[i] * features[i + 1]
                    non_zero += features[-1] * 10
                    epoch.append(ckpt["epoch"])
                    #for k in ckpt["model_state_dict"].keys():
                    #    if "noise" in k or "qz" in k:
                    #        continue
                    #    total += np.prod(ckpt["model_state_dict"][k].size())

                    #ratio_part.append(non_zero_ratio)
                    ratio_list.append(np.log(non_zero))
                    test_acc = ckpt["test_acc"]
                    #acc_part.append(test_acc)
                    test_acc_list.append(test_acc)
                #ratio_list.append(np.mean(ratio_part))
                #test_acc_list.append(np.mean(acc_part))
            #print("sparsity={:.1f}, epochs={}".format(sparsity, np.mean(epoch)))
            plt.scatter(ratio_list, test_acc_list, alpha=0.2, label=label_name)
            idx = np.argsort(0.0 - np.array(test_acc_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))
            idx = np.argsort(np.array(ratio_list))
            for i in idx[:5]:
                print("[{}, sparsity={:.1f}] Test Error= {:.2f}, log(Non-zero)= {:.2f}".format(model_name, sparsity, 100-100*test_acc_list[i], ratio_list[i]))

    plt.legend(loc="lower right")
    plt.axhline(y=0.982, color="black", alpha=0.3, linestyle="--")
    plt.axhline(y=0.984, color="blue", alpha=0.3, linestyle="--")
    plt.axhline(y=0.986, color="red", alpha=0.3, linestyle="--")
    plt.axvline(x=10.19, color="black", alpha=0.3, linestyle="--")
    plt.axvline(x=11.15, color="black", alpha=0.3, linestyle="--")
    plt.title("L0 policy, Results")
    plt.ylabel("Test Acc.")
    plt.xlabel("log(Non-zero)")
    plt.savefig("l0_policy_diff_two_models.png")
    plt.yticks([0.97, 0.982, 0.984, 0.986])
    plt.ylim(0.97, 0.99)
    plt.savefig("l0_policy_diff_two_models_acc_crop.png")
    plt.close()


#summary08()
summary09()
#summary06()
#summary07()

