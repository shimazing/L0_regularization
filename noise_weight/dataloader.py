import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
# Create dummy csv data
nb_samples = 110
a = np.arange(nb_samples)
df = pd.DataFrame(a, columns=['data'])
df.to_csv('data.csv', index=False)


# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples / self.chunksize

    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                skiprows=index * self.chunksize + 1,  #+1, since we skip the header
                chunksize=self.chunksize,
                names=['data']))
        x = torch.from_numpy(x.data.values)
        return x

    def __len__(self):
        return self.len

class ForestCovertype(Dataset):
    def __init__(self, seed, train=True, test_ratio=0.3):
        from sklearn.datasets import fetch_covtype
        X, y = fetch_covtype(return_X_y=True, random_state=seed, shuffle=True)
        y = y - 1 # 0~6
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_ratio)
        if train:
            self.X = torch.from_numpy(train_X)
            self.y = torch.from_numpy(train_y)
        else:
            self.X = torch.from_numpy(test_X)
            self.y = torch.from_numpy(test_y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class Abalone(Dataset):
    def __init__(self, seed, train=True, test_ratio=0.3):
        column_names = ["sex", "length", "diameter", "height", "whole weight",
                        "shucked weight", "viscera weight", "shell weight", "rings"]
        data = pd.read_csv("../data/abalone.data", names=column_names)
        print("Number of samples: %d" % len(data))
        print(data.head())
        for label in "MFI":
            data[label] = (data["sex"] == label)
        del data["sex"]
        y = data.rings.values - 1
        del data["rings"]
        X = data.values.astype(np.float)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_ratio)
        if train:
            self.weight = torch.from_numpy(1. / np.unique(train_y, return_counts=True)[1].astype(float))
            self.X = torch.from_numpy(train_X).float()
            self.y = torch.from_numpy(train_y).long()
        else:
            self.X = torch.from_numpy(test_X).float()
            self.y = torch.from_numpy(test_y).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class RedWine(Dataset):
    def __init__(self, seed, train=True, test_ratio=0.3):
        data = pd.read_csv("../data/winequality-red.csv", sep=';')
        print("Number of samples: %d" % len(data))
        print(data.head())
        y = data.quality.values
        del data["quality"]
        X = data.values.astype(np.float)

        train_X, test_X, train_y, test_y = train_test_split(X, y,
                test_size=test_ratio, random_state=seed)
        if train:
            self.weight = torch.from_numpy(1. / np.unique(train_y,
                return_counts=True)[1].astype(float))
            self.X = torch.from_numpy(train_X).float()
            self.y = torch.from_numpy(train_y).long()
        else:
            self.X = torch.from_numpy(test_X).float()
            self.y = torch.from_numpy(test_y).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class WhiteWine(Dataset):
    def __init__(self, seed, train=True, test_ratio=0.3):
        data = pd.read_csv("../data/winequality-white.csv", sep=';')
        print("Number of samples: %d" % len(data))
        print(data.head())
        y = data['quality'].values - 3
        del data["quality"]
        X = data.values.astype(np.float)

        train_X, test_X, train_y, test_y = train_test_split(X, y,
                test_size=test_ratio, random_state=seed)
        if train:
            self.weight = torch.from_numpy(1. / np.unique(train_y,
                return_counts=True)[1].astype(float))
            print(np.unique(test_y, return_counts=True))
            self.X = torch.from_numpy(train_X).float()
            self.y = torch.from_numpy(train_y).long()
        else:
            self.X = torch.from_numpy(test_X).float()
            self.y = torch.from_numpy(test_y).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    ForestCovertype(0)
    whitewine = WhiteWine(0)
    abalone = Abalone(0)
    print(len(whitewine))
    print(len(abalone))

    '''
    dataset = CSVDataset('data.csv', chunksize=10, nb_samples=nb_samples)
    loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)

    for batch_idx, data in enumerate(loader):
        print('batch: {}\tdata: {}'.format(batch_idx, data))
    '''
