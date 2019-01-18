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
# Load SVMLight files
from sklearn.datasets import load_svmlight_file

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


class LETTER(Dataset):
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

class PROTEIN(Dataset):
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

class SATIMAGE(Dataset):
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

class SHUTTLE(Dataset):
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
