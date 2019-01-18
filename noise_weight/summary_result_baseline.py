import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    model_name_list = ['svm', 'logisticRegression']
    dataset_list = ['ijcnn1']#['wine', 'redwine', 'abalone', 'dna', "letter", "satimage",
            #"ijcnn1",]
    kernel_list = ['rbf', 'linear']
    C_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    rand_seed_list = range(5)

    for model_name in model_name_list:
      for dataset in dataset_list:
          for kernel in kernel_list:
            if model_name == 'logisticRegression' and kernel == 'rbf':
                continue
            best_result(model_name, dataset, kernel, C_list, rand_seed_list)

def best_result(model_name, dataset,kernel, C_list, rand_seed_list):
    CKPT_DIR = "ckpt"
    best_valid_auc = 0
    for C in C_list:
        valid_auc_list = []
        valid_acc_list = []
        test_auc_list = []
        test_acc_list = []
        train_auc_list = []
        train_acc_list = []
        for rand in rand_seed_list:
            ckpt_name = "{}_{}_{}_{}_{}.pth.tar".format(dataset,
                    model_name, C,
                    kernel, rand)
            try:
                ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
            except:
                print(" *** Error opening {}".format(ckpt_name))
                continue
            train_acc_list.append(ckpt['train_acc'])
            train_auc_list.append(ckpt['train_auc'])
            valid_acc_list.append(ckpt['valid_acc'])
            valid_auc_list.append(ckpt['valid_auc'])
            test_acc_list.append(ckpt['test_acc'])
            test_auc_list.append(ckpt['test_auc'])
        train_acc = np.mean(train_acc_list)
        train_auc = np.mean(train_auc_list)
        valid_acc = np.mean(valid_acc_list)
        valid_auc = np.mean(valid_auc_list)
        test_acc = np.mean(test_acc_list)
        test_auc = np.mean(test_auc_list)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_result = (train_auc, valid_auc, test_auc)
            best_C = C
    print(model_name, dataset, kernel,"best C", best_C,"result", best_result)

main()
