import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_svmlight_file
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import os

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
    def __init__(self, train=True, test_ratio=0.3, seed=0):
        column_names = ["sex", "length", "diameter", "height", "whole weight",
                        "shucked weight", "viscera weight", "shell weight", "rings"]
        data = pd.read_csv("../data/abalone.data", names=column_names)
        #print("Number of samples: %d" % len(data))
        #print(data.head())
        valid = [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.n_cls = len(valid)
        #self.n_cls = 28
        self.n_features = 10
        for label in "MFI":
            data[label] = (data["sex"] == label)
        del data["sex"]
        y = data.rings.values
        y[y==29] = 28
        y -= np.min(y)
        del data["rings"]
        X = data.values.astype(np.float)
        from sklearn.preprocessing import StandardScaler
        m = StandardScaler()
        X = m.fit_transform(X)
        valid_bool = (y.reshape(-1, 1) == valid).max(axis=1)
        X = X[valid_bool]
        y = y[valid_bool]
        for i, v in enumerate(valid):
            y[y == v]= i
        assert np.min(y) == 0 and np.max(y) == self.n_cls - 1 and np.unique(y).size == self.n_cls
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_ratio,
                                                            random_state=seed)
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
    def __init__(self, train=True, test_ratio=0.3, seed=0):
        data = pd.read_csv("../data/winequality-red.csv", sep=';')
        #print("Number of samples: %d" % len(data))
        #print(data.head())
        self.n_cls = 6
        self.n_features = 11
        y = data.quality.values
        y -= np.min(y)  # since currently, np.min(y) =3
        del data["quality"]
        X = data.values.astype(np.float)
        from sklearn.preprocessing import StandardScaler
        m = StandardScaler()
        X = m.fit_transform(X)
        #self.n_features = X.shape[1]
        #self.n_cls = np.unique(y).size
        assert np.min(y) == 0 and np.max(y) == self.n_cls - 1 and np.unique(y).size == self.n_cls
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
        from sklearn.preprocessing import StandardScaler
        m = StandardScaler()
        X = m.fit_transform(X)
        self.n_cls = 7
        self.n_features = X.shape[1]
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


class Letter(Dataset):
    def __init__(self, train=True, val=False):
        # # of data: 15,000 / 5,000 (testing) / 10,500 (tr) / 4,500 (val)
        # Source: Statlog / Letter
        self.n_features = 16
        self.n_cls = 26
        if train:
            if not val:
                data = load_svmlight_file("../data/letter.scale.tr", self.n_features)
            else:
                data = load_svmlight_file("../data/letter.scale.val", self.n_features)
        else:
            data = load_svmlight_file("../data/letter.scale.t", self.n_features)

        #check data
        X = data[0].todense()
        y = data[1]
        y = y - np.min(y)
        assert np.min(y) == 0 and np.max(y) == self.n_cls-1 and np.unique(y).size == self.n_cls
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class DNA(Dataset):
    def __init__(self, train=True, val=False):
        # # of data: 2,000 / 1,186 (testing) / 1,400 (tr) / 600 (val)
        # Source: Statlog / Dna
        self.n_features = 180
        self.n_cls = 3
        if train:
            if not val:
                data = load_svmlight_file("../data/dna.scale.tr", self.n_features)
            else:
                data = load_svmlight_file("../data/dna.scale.val", self.n_features)
        else:
            data = load_svmlight_file("../data/dna.scale.t", self.n_features)

        # check data
        X = data[0].todense()
        y = data[1]
        y = y - np.min(y)
        assert np.min(y) == 0 and np.max(y) == self.n_cls-1 and np.unique(y).size == self.n_cls
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class Protein(Dataset):
    def __init__(self, train=True, val=False):
        # Source: [JYW02a]
        # # of data: 17,766 / 6,621 (testing) / 14,895 (tr) / 2,871 (val)
        self.n_features = 357
        self.n_cls = 3
        raise ValueError  # error occur
        if train:
            if not val:
                data = load_svmlight_file("../data/protein.tr", self.n_features)
            else:
                data = load_svmlight_file("../data/protein.val", self.n_features)
        else:
            data = load_svmlight_file("../data/protein.t", self.n_features)

        # check data
        X = data[0].todense()
        y = data[1]
        y = y - np.min(y)
        assert np.min(y) == 0 and np.max(y) == self.n_cls - 1 and np.unique(y).size == self.n_cls
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class Satimage(Dataset):
    def __init__(self, train=True, val=False):
        # Source: Statlog / Satimage
        # Preprocessing: Training data is further separated into two sets, tr and val. [CWH01a]
        # # of data: 4,435 / 2,000 (testing) / 3,104 (tr) / 1,331 (val)
        self.n_cls = 6
        self.n_features = 36
        if train:
            if not val:
                data = load_svmlight_file("../data/satimage.scale.tr", self.n_features)
            else:
                data = load_svmlight_file("../data/satimage.scale.val", self.n_features)
        else:
            data = load_svmlight_file("../data/satimage.scale.t", self.n_features)

        # check data
        X = data[0].todense()
        y = data[1]
        y = y - np.min(y)
        assert np.min(y) == 0 and np.max(y) == self.n_cls - 1 and np.unique(y).size == self.n_cls
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class Shuttle(Dataset):
    def __init__(self, train=True, val=False):
        # Source: Statlog / Shuttle
        # Preprocessing: Training data is further separated into two sets, tr and val. [CWH01a]
        # # of data: 43,500 / 14,500 (testing) / 30,450 (tr) / 13,050 (val)
        raise ValueError  # since 'shuttle.scale.val' does not have label 5
        self.n_cls = 7
        self.n_features = 9
        if train:
            if not val:
                data = load_svmlight_file("../data/shuttle.scale.tr", self.n_features)
            else:
                data = load_svmlight_file("../data/shuttle.scale.val", self.n_features)
        else:
            data = load_svmlight_file("../data/shuttle.scale.t", self.n_features)

        # check data
        X = data[0].todense()
        y = data[1]
        y = y - np.min(y)
        assert np.min(y) == 0 and np.max(y) == self.n_cls - 1 and np.unique(y).size == self.n_cls
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class IJCNN1(Dataset):
    def __init__(self, train=True, val=False):
        # Source: [DP01a]
        # Preprocessing: We use winner's transformation [Chang01d]
        # # of data: 49,990 / 91,701 (testing)
        self.n_cls = 2
        self.n_features = 22
        if train:
            if not val:
                data = load_svmlight_file("../data/ijcnn1.tr", self.n_features)
            else:
                data = load_svmlight_file("../data/ijcnn1.val", self.n_features)
        else:
            data = load_svmlight_file("../data/ijcnn1.t", self.n_features)

        # check data
        X = data[0].todense()
        y = (data[1]+1)/2 # Original y has {-1,1}, convert {-1,1} --> {0, 1}
        y = y - np.min(y)
        assert np.min(y) == 0 and np.max(y) == self.n_cls - 1 and np.unique(y).size == self.n_cls
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

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


def load_data(dataset, args, kwargs, rand_seed=0):
    if dataset == "letter":
        train_set = Letter(train=True, val=False)
        valid_set = Letter(train=True, val=True)
        test_set = Letter(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif dataset == "dna":
        train_set = DNA(train=True, val=False)
        valid_set = DNA(train=True, val=True)
        test_set = DNA(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif dataset == "satimage":
        train_set = Satimage(train=True, val=False)
        valid_set = Satimage(train=True, val=True)
        test_set = Satimage(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif dataset == "shuttle":
        train_set = Shuttle(train=True, val=False)
        valid_set = Shuttle(train=True, val=True)
        test_set = Shuttle(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif dataset == "ijcnn1":
        train_set = IJCNN1(train=True, val=False)
        valid_set = IJCNN1(train=True, val=True)
        test_set = IJCNN1(train=False)
        n_features = test_set.n_features
        n_cls = test_set.n_cls
    elif dataset == "mnist":
        normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        n_features = 784
        n_cls = 10
        dset_string = "datasets.MNIST"
        train_tfms = []
        train_tfms += [transforms.ToTensor(), normalize]
        train_set = eval(dset_string)(root=DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)

        valid_set = eval(dset_string)(root=DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)

        test_set = eval(dset_string)(root=DATA_DIR, train=False,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)
    elif dataset == "fashionmnist":
        dset_string = "datasets.FashionMNIST"
        n_features = 784
        n_cls = 10
        train_tfms = []
        train_tfms += [transforms.ToTensor()]#, normalize]
        Fashion_DATA_DIR = os.path.join(DATA_DIR, "FashionMNIST")
        train_set = eval(dset_string)(root=Fashion_DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)

        valid_set = eval(dset_string)(root=Fashion_DATA_DIR, train=True,
                                      transform=transforms.Compose(train_tfms), download=True)

        test_set = eval(dset_string)(root=Fashion_DATA_DIR, train=False,
                                     transform=transforms.Compose([transforms.ToTensor()]), download=True)
    elif dataset == "svhn":
        input_channel = 3
        n_features = 3*32*32
        n_cls = 10
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
        train_tfms = [transforms.ToTensor(), normalize]
        dset_string = "datasets.SVHN"
        svhn_DATA_DIR = os.path.join(DATA_DIR, "SVHN")
        train_set = eval(dset_string)(root=svhn_DATA_DIR, split="train", transform=transforms.Compose(train_tfms), download=True)
        valid_set = eval(dset_string)(root=svhn_DATA_DIR, split="train", transform=transforms.Compose(train_tfms), download=True)
        test_set = eval(dset_string)(root=svhn_DATA_DIR, split="test", transform=transforms.Compose(train_tfms), download=True)
    elif "cifar" in dataset:
        n_features = 3 * 32 * 32
        n_cls = 10 if dataset.endswith("cifar10") else 100
        dset_string = 'datasets.CIFAR10' if dataset.endswith("cifar10") else 'datasets.CIFAR100'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_tfms = [transforms.ToTensor(), normalize]
        if "preprocessed" in dataset:
            print(dataset)
            train_tfms = [transforms.RandomHorizontalFlip()] + train_tfms
            train_tfms = [transforms.RandomCrop(32, 4)] + train_tfms
        train_set = eval(dset_string)(root='../data', train=True, transform=transforms.Compose(train_tfms),
                                      download=True)
        valid_set = eval(dset_string)(root='../data', train=True, transform=transforms.Compose(train_tfms),
                                      download=True)
        test_set = eval(dset_string)(root='../data', train=False,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]),
                                     download=True)
    elif dataset == "redwine":
        train_set = RedWine(train=True)
        valid_set = RedWine(train=True)
        test_set = RedWine(train=False)
        n_cls = test_set.n_cls
        n_features = test_set.n_features
    elif dataset == "whitewine":
        train_set = WhiteWine(rand_seed, train=True)
        valid_set = WhiteWine(rand_seed, train=True)
        test_set = WhiteWine(rand_seed, train=False)
        n_cls = test_set.n_cls
        n_features = test_set.n_features
    elif dataset == "abalone":
        train_set = Abalone(train=True)
        valid_set = Abalone(train=True)
        test_set = Abalone(train=False)
        n_cls = test_set.n_cls
        n_features = test_set.n_features
    else:
        raise ValueError

    if dataset in ["mnist","preprocessed-cifar10", "preprocessed-cifar100",
            "whitewine", "redwine", "abalone", "redwine"]:
        # For those data where validation set is not given
        num_train = len(train_set)
        indices = list(range(num_train))
        valid_size = 0.25
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   sampler=train_sampler, **kwargs)
        valid_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   sampler=valid_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif dataset in ['cifar10', 'cifar100', 'fashionmnist', 'svhn']:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader =  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif dataset in ["letter", "dna", "satimage", "shuttle", "ijcnn1"]:
        # For those data where validation set is given
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
                                                   shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, **kwargs)
    else:
        raise ValueError
    return train_loader, valid_loader, test_loader, n_cls, n_features
