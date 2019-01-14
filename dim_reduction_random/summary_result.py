import os
import torch
import numpy as np

POLICY = "Only-Output-layer-Rank-Control"


model_name_list = ["MLP-300-100"]
rank_list = [100, 80, 60, 40]
#model_name_list = ["MLP-50-30", "MLP-100-50", "MLP-300-100",
#        "MLP-500-300", "MLP-1000-500", "MLP-2000-1000"]
SEED_MAX =10
CKPT_DIR = "ckpt"

for model_name in model_name_list:
  for rank in rank_list:
    acc_part = []
    for rand_seed in range(SEED_MAX):
        try:
            ckpt_name = "best_{}_{}_{}_{}.pth.tar".format(model_name, POLICY, rank, rand_seed)
            ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name))
        except Exception as e:
            print(e)
            continue
        test_acc = ckpt["test_acc"]
        acc_part.append(test_acc)

    print("[{} {}] Test Acc: {:.2f}".format(model_name, rank, np.mean(acc_part)*100))

