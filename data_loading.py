import os

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import numpy as np
import cv2


class FloatXFloatYTensorDataset(Dataset):
    """
    Simple dataset with FloatTensor format of X and FloatTensor format of Y.
    Float type is numpy.float32
    """

    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__()
        self.y = torch.FloatTensor(np.float32(y))
        self.X = torch.FloatTensor(np.float32(X))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx, :]
        label = self.y[idx]
        if self.transform:
            image = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label


class FloatXLongYTensorDataset(Dataset):
    """
    Simple dataset with FloatTensor format of X and LongTensor format of Y.
    Float type is numpy.float32, Long is numpy.long()
    """

    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__()
        self.y = torch.LongTensor(np.long(y))
        self.X = torch.FloatTensor(np.float32(X))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx, :]
        label = self.y[idx]
        if self.transform:
            image = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label


class GrayImageDataset(Dataset):
    """
    class to create dataset for the further DataLoaders
    """

    def __init__(self, data_directory, label_extraction_func, size_tuple, transform=None, target_transform=None):
        self.data_directory = data_directory
        self.image_filenames = os.listdir(data_directory)
        self.label_extractor = label_extraction_func
        self.size = size_tuple
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.data_directory, filename)
        label = self.label_extractor(filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image / 255
        return image, label


def simple_dataloader_train_test(train_dataset: Dataset, test_dataset: Dataset, batch_sizes_tuple: tuple = (32, 32),
                                 num_workers: int = 1) -> dict:
    train_dl = DataLoader(train_dataset, batch_size=batch_sizes_tuple[1], shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test_dataset, batch_size=batch_sizes_tuple[1], shuffle=False, num_workers=num_workers)
    return {'train': train_dl, 'test': test_dl}


def weighted_from_y_dataloader_train_test(train_dataset: Dataset, test_dataset: Dataset, y_train,
                                          batch_sizes_tuple: tuple = (32, 32),
                                          num_workers: int = 1) -> dict:
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_dl = DataLoader(train_dataset, batch_sizes_tuple[0], sampler=sampler, num_workers=num_workers)
    test_dl = DataLoader(test_dataset, batch_sizes_tuple[1], shuffle=False, num_workers=num_workers)
    return {'train': train_dl, 'test': test_dl}
