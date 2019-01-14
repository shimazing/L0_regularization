import os
import torch
import numpy as np

POLICY = "Only-Output-layer"


model_name_list = ["MLP-50-30", "MLP-100-50", "MLP-300-100",
        "MLP-500-300", "MLP-1000-500", "MLP-2000-1000"]
SEED_MAX =20
CKPT_DIR = "ckpt"

for model_name in model_name_list:
    acc_part = []
    for rand_seed in range(SEED_MAX):
        ckpt_name = "best_{}_{}_{}.pth.tar".format(model_name, POLICY, rand_seed)
        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
        test_acc = ckpt["test_acc"]
        acc_part.append(test_acc)

    print("[{}] Test Acc: {:.2f}".format(model_name, np.mean(acc_part)*100))

