import os
import torch
import numpy as np
import matplotlib
import argparse
import matplotlib.pyplot as plt

from matplotlib import rcParams
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
          'figure.autolayout': True}
rcParams.update(params)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="fashionmnist")#,required=True)
parser.add_argument("--act_fn", type=str, default="relu")#, required=True)
args = parser.parse_args()

POLICY_LIST = ["IntermediateNoisyMLP"]#, "OutgoingNoisyMLP"]#, "IncomingNoisyMLP"]
SEED_MAX = 2
MARKER_SIZE = 60
FONT_SIZE = 18


def main():
    for nhid in [10, 50, 100, 500]:
        for learnt_layer in [2, 4]:
            #time_check([nhid], learnt_layer)
            test_acc_comprehensive([nhid],learnt_layer)

            #best_nlayer_dropoutintermediate_mlp([nhid])
            #learning_acc([nhid], learnt_layer)
            #learning_loss([nhid], learnt_layer)


def time_check(nhid_list, learnt_layer, n_layer_list=(6,8,10)):
    CKPT_DIR = "ckpt"
    SEED_MAX = 1
    fix, ax = plt.subplots(1, 1)#, figsize=(13, 13))
    policy_list = ["IntermediateNoisyMLP"]
    policy_list = POLICY_LIST
    forward_time_list = []
    backward_time_list = []
    step_time_list = []
    policy_label = []
    for policy in policy_list:
        for n_layer in n_layer_list:
            if policy == "IncomingNoisyMLP":
                noise_layer_list = [-1]  # , n_layer-learnt_layer-1] # Here, -1 means fully training
            elif policy == "OutgoingNoisyMLP":
                # if policy == "OutgoingNoisyMLP":
                noise_layer_list = [learnt_layer]
            elif policy == "IntermediateNoisyMLP":
                assert learnt_layer % 2 == 0
                freezing_start = int(learnt_layer / 2)
                freezing_end = n_layer - int(learnt_layer / 2) - 1
                noise_layer_list = ["{}{}".format(freezing_start, freezing_end)]
            elif policy == "AlternatingNoisyMLP":
                noise_layer_list = [np.ceil(n_layer / learnt_layer).astype(int)]
            else:
                raise ValueError
            for noise_layer in noise_layer_list:
                for rand_seed in range(SEED_MAX):
                    ckpt_name = "{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                            nhid_list[0], n_layer, policy, noise_layer,
                            args.act_fn, rand_seed)
                    try:
                        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    except:
                        print(" *** Error opening {}".format(ckpt_name))
                        continue
                    forward_time_list.append(np.mean(ckpt["forward_time_list"]))
                    backward_time_list.append(np.mean(ckpt["backward_time_list"]))
                    step_time_list.append(np.mean(ckpt["step_time_list"]))
                    color, marker, label_name = get_graph_info(policy, noise_layer, n_layer)
                    policy_label.append(label_name)

    add_list = [learnt_layer] if learnt_layer==4 else [learnt_layer, learnt_layer+2]
    for nlayer in add_list + list(n_layer_list):
        base_model_name_list = gen_model_name_list(nhid_list, nlayer)
        base_policy_list = ["IncomingNoisyMLP"]
        for model_name in base_model_name_list:
            noise_layer = -1
            for policy in base_policy_list:
                for rand_seed in range(SEED_MAX):
                    ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name, policy, noise_layer,
                                                                   args.act_fn, rand_seed)
                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        ckpt_name = "{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                                nhid_list[0], nlayer, policy, noise_layer,
                                args.act_fn, rand_seed)

                    try:
                        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    except:
                        print(" *** Error opening {}".format(ckpt_name))
                        continue
                    forward_time_list.append(np.mean(ckpt["forward_time_list"]))
                    backward_time_list.append(np.mean(ckpt["backward_time_list"]))
                    step_time_list.append(np.mean(ckpt["step_time_list"]))
                    color, marker, label_name = get_graph_info(policy, noise_layer, nlayer)
                    policy_label.append(label_name)

    width = 0.35
    noise_layer_list = range(len(policy_label))
    #plt.barh(noise_layer_list, step_time_list, width, label='optim')
    lefts = np.array(step_time_list)+np.array(backward_time_list)
    #plt.barh(noise_layer_list, lefts, width, label='backward')
    #plt.bar(noise_layer_list, backward_time_list, width, label='backward',
    #       bottom=step_time_list)
    #plt.barh(noise_layer_list, forward_time_list, width, label='forward',
    #       bottom=[x + y for x, y in zip(step_time_list,
    #                                     backward_time_list)])
    lefts = lefts + np.array(forward_time_list)
    plt.barh(noise_layer_list, lefts, width, label='forward')
    lefts -= np.array(forward_time_list)
    plt.barh(noise_layer_list, lefts, width, label='backward')

    plt.barh(noise_layer_list, step_time_list, width, label='update')
    plt.legend()
    #plt.tight_layout()
    plt.yticks(noise_layer_list, policy_label)
    plt.xlabel("Taken Training Time for One Epoch")
    plt.show()
    plt.savefig("{}_FrozenMLPs_Time_{}_{}_{}.png".format(args.dataset,nhid_list[0],learnt_layer,args.act_fn))
    plt.close()

def learning_acc(nhid_list, learnt_layer, n_layer_list=(6,8,10)):
    CKPT_DIR = "ckpt"
    fix, ax = plt.subplots(1, 1, figsize=(10, 10))
    first_noise_layer = 2
    policy_list = ["IntermediateNoisyMLP", "OutgoingNoisyMLP"]
    policy_list = POLICY_LIST
    for n_layer in n_layer_list:
        model_name_list = gen_model_name_list(nhid_list, n_layer)
        #noise_layer_list = ["{}{}".format(first_noise_layer, nlayer - 3)] # "1{}".format(nlayer-2)]
        for model_name in model_name_list:
            for policy in policy_list:
                if policy == "IncomingNoisyMLP":
                    noise_layer_list = [-1]  # , n_layer-learnt_layer-1] # Here, -1 means fully training
                elif policy == "OutgoingNoisyMLP":
                    # if policy == "OutgoingNoisyMLP":
                    noise_layer_list = [learnt_layer]
                elif policy == "IntermediateNoisyMLP":
                    assert learnt_layer % 2 == 0
                    freezing_start = int(learnt_layer / 2)
                    freezing_end = n_layer - int(learnt_layer / 2) - 1
                    noise_layer_list = ["{}{}".format(freezing_start, freezing_end)]
                elif policy == "AlternatingNoisyMLP":
                    noise_layer_list = [np.ceil(n_layer / learnt_layer).astype(int)]
                else:
                    raise ValueError
                for noise_layer in noise_layer_list:
                    acc_part = []
                    train_acc_list = []
                    valid_acc_list = []
                    test_acc_list = []
                    non_zero_list = []
                    test_acc_list_ = []
                    for rand_seed in range(SEED_MAX):
                        ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name,
                                                                       policy, noise_layer, args.act_fn,
                                                                       rand_seed)
                        if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                            ckpt_name = "{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                                    nhid_list[0], n_layer,policy, noise_layer,
                                    args.act_fn, rand_seed)
                        try:
                            ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                        except:
                            print(" *** Error opening {}".format(ckpt_name))
                            continue
                        test_acc = ckpt["test_auc"]
                        acc_part.append(test_acc)
                        train_acc_list.append(ckpt["train_auc_list"])
                        valid_acc_list.append(ckpt["valid_auc_list"])

                    color, marker, label_name = get_graph_info(policy, noise_layer, n_layer)
                    ax.plot(np.mean(np.array(train_acc_list),axis=0), color=color, alpha=0.5,
                            label=label_name+", train")
                    ax.plot(np.mean(np.array(valid_acc_list),axis=0), color=color, alpha=0.8,
                            linestyle="--", label=label_name+", test")

    base_model_name_list = gen_model_name_list(nhid_list, learnt_layer)
    base_policy_list = ["IncomingNoisyMLP"] #, "DropoutMLP"]
    for model_name in base_model_name_list:
        noise_layer = -1
        for policy in base_policy_list:
            train_acc_list = []
            valid_acc_list = []
            for rand_seed in range(SEED_MAX):
                ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name, policy, noise_layer,
                                                               args.act_fn, rand_seed)

                if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                    ckpt_name = "{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                            nhid_list[0], learnt_layer, policy, noise_layer,
                            args.act_fn, rand_seed)
                try:
                    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                except:
                    print(" *** Error opening {}".format(ckpt_name))
                    continue
                train_acc_list.append(ckpt["train_auc_list"])
                valid_acc_list.append(ckpt["valid_auc_list"])


            color, marker, label_name = get_graph_info(policy, noise_layer, first_noise_layer*2)
            ax.plot(np.mean(np.array(train_acc_list),axis=0), color=color, alpha=0.5,
                    label=label_name + ", train")
            ax.plot(np.mean(np.array(valid_acc_list),axis=0), color=color, alpha=0.8,
                    linestyle="--", label=label_name + ", test")
    ax.legend()#loc="lower right")#, prop={'size': 30})
    ax.set_ylabel("Acc")
    ax.set_xlabel("Epoch")
    ax.set_title("[{}] Frozen MLPs".format(args.dataset.upper()))
    plt.show()
    plt.savefig("{}_learning_acc_FrozenMLPs_{}_{}_{}.png".format(args.dataset,nhid_list[0],learnt_layer,args.act_fn))
    plt.close()

def learning_loss(nhid_list, learnt_layer, n_layer_list=(6,8,10)):
    CKPT_DIR = "ckpt"
    fix, ax = plt.subplots(1, 1, figsize=(10, 10))
    first_noise_layer = 2
    policy_list = ["IntermediateNoisyMLP", "OutgoingNoisyMLP"]
    policy_list = POLICY_LIST
    for n_layer in n_layer_list:
        model_name_list = gen_model_name_list(nhid_list, n_layer)
        for model_name in model_name_list:
            for policy in policy_list:
                if policy == "IncomingNoisyMLP":
                    noise_layer_list = [-1]  # , n_layer-learnt_layer-1] # Here, -1 means fully training
                elif policy == "OutgoingNoisyMLP":
                    # if policy == "OutgoingNoisyMLP":
                    noise_layer_list = [learnt_layer]
                elif policy == "IntermediateNoisyMLP":
                    assert learnt_layer % 2 == 0
                    freezing_start = int(learnt_layer / 2)
                    freezing_end = n_layer - int(learnt_layer / 2) - 1
                    noise_layer_list = ["{}{}".format(freezing_start, freezing_end)]
                elif policy == "AlternatingNoisyMLP":
                    noise_layer_list = [np.ceil(n_layer / learnt_layer).astype(int)]
                else:
                    raise ValueError
                for noise_layer in noise_layer_list:
                    acc_part = []
                    train_acc_list = []
                    valid_acc_list = []
                    test_acc_list = []
                    non_zero_list = []
                    test_acc_list_ = []
                    for rand_seed in range(SEED_MAX):
                        ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name,
                                                                       policy, noise_layer, args.act_fn,
                                                                       rand_seed)
                        if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                            ckpt_name = "{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                                    nhid_list[0], n_layer, policy, noise_layer,
                                    args.act_fn, rand_seed)
                        try:
                            ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                        except:
                            print(" *** Error opening {}".format(ckpt_name))
                            continue
                        test_acc = ckpt["test_auc"]
                        acc_part.append(test_auc)
                        train_acc_list.append(ckpt["train_loss_list"])
                        valid_acc_list.append(ckpt["valid_loss_list"])

                    color, marker, label_name = get_graph_info(policy, noise_layer, n_layer)
                    ax.plot(np.mean(np.array(train_acc_list),axis=0), color=color, alpha=0.5,
                            label=label_name+", train")
                    ax.plot(np.mean(np.array(valid_acc_list),axis=0), color=color, alpha=0.8,
                            linestyle="--", label=label_name+", test")

    for nlayer in [learnt_layer, learnt_layer*2] + list(n_layer_list):
        base_model_name_list = gen_model_name_list(nhid_list, nlayer)
        base_policy_list = ["IncomingNoisyMLP"]#, "DropoutMLP"]
        for model_name in base_model_name_list:
            noise_layer = -1
            for policy in base_policy_list:
                train_acc_list = []
                valid_acc_list = []
                for rand_seed in range(SEED_MAX):
                    ckpt_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name, policy, noise_layer,
                                                                   args.act_fn, rand_seed)

                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        ckpt_name = "{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                                nhid_list[0], nlayer,policy, noise_layer,
                                args.act_fn, rand_seed)
                    try:
                        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    except:
                        print(" *** Error opening {}".format(ckpt_name))
                        continue
                    train_acc_list.append(ckpt["train_loss_list"])
                    valid_acc_list.append(ckpt["valid_loss_list"])


                color, marker, label_name = get_graph_info(policy, noise_layer, nlayer)
                ax.plot(np.mean(np.array(train_acc_list),axis=0), color=color, alpha=0.5,
                        label=label_name + ", train")
                ax.plot(np.mean(np.array(valid_acc_list),axis=0), color=color, alpha=0.8,
                        linestyle="--", label=label_name + ", valid")
        ax.legend()#loc="lower right")#, prop={'size': 30})
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("[{}] Frozen MLPs".format(args.dataset.upper()))
    plt.show()
    plt.savefig("{}_learning_loss_FrozenMLPs_{}_{}_{}.png".format(args.dataset,nhid_list[0],learnt_layer,args.act_fn))
    plt.close()

def get_graph_info(policy, noise_layer, n_layer):
    if policy == "OutgoingNoisyMLP":
        if noise_layer == 1:
            color = "r"
            marker = "P"
        elif noise_layer == 2:
            color = "g"
            if n_layer == 4:
                marker = ">"
            elif n_layer == 6:
                marker = "P"
            elif n_layer == 8:
                marker = "D"
            elif n_layer == 10:
                marker = "<"
        elif noise_layer == 3:
            color = "b"
            marker = "s"
        elif noise_layer == 4:
            color = "r"
            if n_layer == 4:
                marker = ">"
            elif n_layer == 6:
                marker = "P"
            elif n_layer == 8:
                marker = "D"
            elif n_layer == 10:
                marker = "<"
        else:
            print("Out noise", noise_layer)
            raise ValueError
        label_name = "Learnt {}, (Top) Frozen {}".format(noise_layer, n_layer-noise_layer)
    elif policy == "IncomingNoisyMLP":
        if noise_layer == -1:
            color = "black"
            if n_layer == 2:
                marker = "o"
            elif n_layer == 4:
                marker = ">"
            elif n_layer == 6:
                marker = "P"
            elif n_layer == 8:
                marker = "D"
            elif n_layer == 10:
                marker = "<"
            label_name = "Learnt {}, No Frozen".format(n_layer)
        else:
            raise ValueError
    elif policy == "IntermediateNoisyMLP":
        if noise_layer.startswith("1"):
            color = "b"
            if n_layer == 4:
                marker = ">"
            elif n_layer == 6:
                marker = "P"
            elif n_layer == 8:
                marker = "D"
            elif n_layer == 10:
                marker = "<"
            learnt_layer = int(noise_layer[1]) - int(noise_layer[0])+1
        elif noise_layer.startswith("2"):
            color = "m"
            if n_layer == 4:
                marker = ">"
            elif n_layer == 6:
                marker = "P"
            elif n_layer == 8:
                marker = "D"
            elif n_layer == 10:
                marker = "<"
            learnt_layer = int(noise_layer[1]) - int(noise_layer[0])+1
        else:
            raise ValueError
        label_name = "Learnt {}, (Middle) Frozen {}".format(n_layer-learnt_layer, learnt_layer)
    else:
        raise ValueError
    return color, marker, label_name

def test_acc_comprehensive(nhid_list, learnt_layer, nlayer_list=(6,8,10)):
    CKPT_DIR = "ckpt"
    # nhid_list = [100, 500, 1000]
    # n_layer = 3
    #plt.style.use("ggplot")
    fix, ax = plt.subplots(1, 1, figsize=(8, 8))
    policy_list = ["IntermediateNoisyMLP"]#, "DropoutIntermediateNoisyMLP2"]
    policy_list = POLICY_LIST

    for policy in policy_list:
        #model_name_list = gen_model_name_list(nhid_list, nlayer)
        #print(model_name_list)
        #first_noise_layer = 2
        #noise_layer_list = ["{}{}".format(first_noise_layer, nlayer - 3)] # "1{}".format(nlayer-2)]
        for nlayer in nlayer_list:
            if policy == "IncomingNoisyMLP":
                noise_layer_list = [-1]  # , n_layer-learnt_layer-1] # Here, -1 means fully training
            elif policy == "OutgoingNoisyMLP":
                # if policy == "OutgoingNoisyMLP":
                noise_layer_list = [learnt_layer]
            elif policy == "IntermediateNoisyMLP":
                assert learnt_layer % 2 == 0
                freezing_start = int(learnt_layer / 2)
                freezing_end = nlayer - int(learnt_layer / 2) - 1
                noise_layer_list = ["{}{}".format(freezing_start, freezing_end)]
            elif policy == "AlternatingNoisyMLP":
                noise_layer_list = [np.ceil(nlayer / learnt_layer).astype(int)]
            else:
                raise ValueError
            for noise_layer in noise_layer_list:
                acc_part = []
                test_acc_list = []
                non_zero_list = []
                for rand_seed in range(SEED_MAX):
                    ckpt_name = "best_{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                            nhid_list[0], nlayer,policy, noise_layer,
                            args.act_fn, rand_seed)
                    try:
                        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    except:
                        print(" *** Error opening {}".format(ckpt_name))
                        continue
                    test_acc = ckpt["test_auc"]
                    acc_part.append(test_acc)

                    #print("[{}] forward_time={:.2f}, backward_time={:.2f}, step_time={:.2f}".format(ckpt_name, np.mean(ckpt["forward_time_list"]),
                     #   np.mean(ckpt["backward_time_list"]), np.mean(ckpt["step_time_list"])))
                print("[(best){}_{}_{}_{}_{}_{}] Test Acc: {:.2f}".format(args.dataset, policy, "MLP",
                    nhid_list[0], nlayer,
                                                                 noise_layer, np.mean(acc_part)*100))
                test_acc_list.append(np.mean(acc_part))
                non_zero_list.append(ckpt["non_zero_list"][-1])

                color, marker, label_name = get_graph_info(policy, noise_layer, nlayer)
                ax.scatter(np.log(non_zero_list), test_acc_list, alpha=0.8, color=color,
                           marker=marker, s=MARKER_SIZE, label=label_name)

    add_list = [learnt_layer] if learnt_layer==4 else [learnt_layer, learnt_layer+2]
    for nlayer in add_list + list(nlayer_list):
        base_model_name_list = gen_model_name_list(nhid_list, nlayer)
        base_policy_list = ["IncomingNoisyMLP"]#, "DropoutMLP"]
        for model_name in base_model_name_list:
            noise_layer = -1
            for policy in base_policy_list:
                acc_part = []
                for rand_seed in range(SEED_MAX):
                    ckpt_name = "best_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, model_name, policy, noise_layer,
                                                                   args.act_fn, rand_seed)
                    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
                        ckpt_name = "best_{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(args.dataset, "MLP",
                                nhid_list[0], nlayer, policy, noise_layer,
                                args.act_fn, rand_seed)

                    try:
                        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
                    except:
                        print(" *** Error opening {}".format(ckpt_name))
                        continue
                    acc_part.append(ckpt["test_auc"])
                color, marker, label_name = get_graph_info(policy, noise_layer, nlayer)
                ax.scatter(np.log(ckpt["non_zero_list"][-1]), np.mean(acc_part), color=color, alpha=0.8,
                           marker=marker, s=MARKER_SIZE, label=label_name)

                print("[(best){}_{}_{}_{}] Test Acc: {:.2f}".format(args.dataset, policy, model_name,
                                                                    noise_layer, np.mean(acc_part) * 100))

    ax.legend()#loc="lower right")#, prop={'size': 30})
    ax.set_ylabel("Test Acc", fontsize='xx-large')
    ax.set_xlabel("log(Number of Trained Params)", fontsize='xx-large')
    ax.set_title("[{}] Frozen MLPs".format(args.dataset.upper()))
    plt.show()
    plt.savefig("{}_best_FrozenMLPs_comprehensive_{}_{}_{}.png".format(args.dataset,nhid_list[0],
        learnt_layer, args.act_fn))
    plt.close()



def gen_model_name_list(nhid_list, n_layer):
    model_name_list = []
    for nhid in nhid_list:
        model_name_list.append("MLP" + "-{}".format(nhid) * n_layer)
    return model_name_list


main()
