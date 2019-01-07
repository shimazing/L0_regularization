from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
import torch.utils.data
import numpy as np
from PIL import Image

class pMNIST(MNIST):
    def __init__(self,
                 n_tasks=100,
                 perms=None,
                 root='./data',
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 target_transform=None,
                 train=True,
                 download=True):

        super(pMNIST, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)
        data_dim = 784
        np.random.seed(1234)
        if perms is None:
            assert n_tasks is not None
            perms = [np.arange(data_dim-1,-1,-1), np.arange(data_dim)] + \
                    [np.random.permutation(data_dim) for _ in range(n_tasks)]
            #perms = [np.arange(data_dim)]
            #for _ in range(n_tasks-1):
            #    perms.append(np.random.permutation(data_dim))
        else:
            if n_tasks is None:
                n_tasks = len(perms)
            assert n_tasks == len(perms)
            #assert np.all(perms[0] == np.arange(data_dim))

        self.perms = perms
        self.task_id = None

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # img pixel permutation
        img = img.view(-1)
        #img = img.index_select(1, torch.LongTensor(self.perms[self.task_id]))
        return img, target

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_id(self):
        return self.task_id

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class iCIFAR10(CIFAR10):
    def __init__(self,
                 classes,
                 root='./data',
                 transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                    (0.2023, 0.1994, 0.2010))]),
                 target_transform=None,
                 train=True,
                 download=True):
        '''
        CIFAR10 for incremental learning with selected classes
        :param root: directory which data will be stored in
        :param classes: which classes are selected for this dataset
        :param transform: transform input image data
        :param target_transform: transform class of image (usually normailzed transformation used)
        :param train: training or not
        :param download: downloading or not
        '''
        assert len(classes) <= 10
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.classes = classes
        # Select subset of classes
        print('[iCIFAR10] create data stream for classes: ', classes)
        if self.train:
            train_data = []
            train_labels = []
            for ii in np.arange(len(self.train_data)):
                if self.train_labels[ii] in self.classes:
                    train_data.append(self.train_data[ii])
                    train_labels.append(self.train_labels[ii])
            # convert data and labels as numpy array
            self.train_data = np.array(train_data)
            self.train_labels = np.array(train_labels)
            print('Creating training data is Done.')
        else:
            test_data = []
            test_labels = []
            for ii in np.arange(len(self.test_data)):
                if self.test_labels[ii] in classes:
                    test_data.append(self.test_data[ii])
                    test_labels.append(self.test_labels[ii])
            self.test_data = np.array(test_data)
            self.test_labels = np.array(test_labels)
            print('Creating testing data is Done.')

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        target -= min(self.classes)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_of_class(self, label):
        if self.train:
            return self.train_data[self.train_labels == label]
        else:
            return self.test_data[self.test_labels == label]

    def get_k_images_of_class(self, label, n_images):
        if self.train:
            images = self.train_data[self.train_labels == label]
        else:
            images = self.test_data[self.test_labels == label]
        images_index = np.arange(len(images))
        if len(images) > n_images:
            np.random.shuffle(images_index)
            return images[images_index[:n_images]], n_images
        else:
            return images, len(images)


    def append(self, images, labels):
        '''
        Append dataset with given images and labels
        :param images: Tensor of shape (N, C, H, W)
        N batch size, C color channel, H height of image, W width of image
        :param labels: list of labels
        :return:
        '''
        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = np.append(self.train_labels, labels)

class iCIFAR100(CIFAR100):
    def __init__(self,
                 classes,
                 root='./data',
                 transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                    (0.2023, 0.1994, 0.2010))]),
                 target_transform=None,
                 train=True,
                 download=True):
        '''
        CIFAR100 for incremental learning with selected classes
        :param root: directory which data will be stored in
        :param classes: which classes are selected for this dataset
        :param transform: transform input image data
        :param target_transform: transform class of image (usually normailzed transformation used)
        :param train: training or not
        :param download: downloading or not
        '''
        assert len(classes) <= 100
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.classes = classes
        # Select subset of classes
        print('[iCIFAR100] create data stream for classes: ', classes)
        if self.train:
            train_data = []
            train_labels = []
            for ii in np.arange(len(self.train_data)):
                if self.train_labels[ii] in self.classes:
                    train_data.append(self.train_data[ii])
                    train_labels.append(self.train_labels[ii])
            # convert data and labels as numpy array
            self.train_data = np.array(train_data)
            self.train_labels = np.array(train_labels)
            print('Creating training data is Done.')
        else:
            test_data = []
            test_labels = []
            for ii in np.arange(len(self.test_data)):
                if self.test_labels[ii] in classes:
                    test_data.append(self.test_data[ii])
                    test_labels.append(self.test_labels[ii])
            self.test_data = np.array(test_data)
            self.test_labels = np.array(test_labels)
            print('Creating testing data is Done.')

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        target -= min(self.classes)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_k_images_of_class(self, label, n_images):
        if self.train:
            images = self.train_data[self.train_labels == label]
        else:
            images = self.test_data[self.test_labels == label]
        images_index = np.arange(len(images))
        if len(images) > n_images:
            np.random.shuffle(images_index)
            return images[images_index[:n_images]], n_images
        else:
            return images, len(images)

    def get_image_of_class(self, label):
        if self.train:
            return self.train_data[self.train_labels == label]
        else:
            return self.test_data[self.test_labels == label]

    def append(self, images, labels):
        '''
        Append dataset with given images and labels
        :param images: Tensor of shape (N, C, H, W)
        N batch size, C color channel, H height of image, W width of image
        :param labels: list of labels
        :return:
        '''
        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = np.append(self.train_labels, labels)

class iIMAGENET(object):
    pass

def data_loader(data_name, train_classes, test_classes, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if data_name == 'cifar10':
        train_set = iCIFAR10(classes=train_classes)
        test_set = iCIFAR10(classes=test_classes,
                            train=False)
    elif data_name == 'cifar100':
        train_set = iCIFAR100(classes=train_classes)
        test_set = iCIFAR100(classes=test_classes,
                             train=False)
    else:
        assert False, 'Not yet supported dataset, {}'.format(data_name)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True, **kwargs)
    return train_set, train_loader, test_set, test_loader
