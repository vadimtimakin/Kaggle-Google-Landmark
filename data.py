import numpy as np
import pandas as pd

import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from custom.augmentations import *

from config import config


class ClassificationDataset(Dataset):
    """The Dataset class."""
    def __init__(self, config, df, transforms=None):
        self.config = config
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # sample_id = self.df[self.config.data.id_column].iloc[idx] + self.config.data.image_format
        sample_id = self.df[self.config.data.id_column].iloc[idx]

        # if not self.config.inference.inference:
        #     sample_path = os.path.join(self.config.paths.path_to_images, sample_id + self.config.data.image_format)
        # else:
        #     sample_path = os.path.join(self.config.paths.path_to_inference_images, sample_id + self.config.data.image_format)
        sample_path = sample_id
        image = cv2.imread(sample_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)['image']

        if not self.config.inference.inference:
            label = torch.tensor(self.df[self.config.data.target_columns].iloc[idx])
        else:
            return image

        return image, label


def get_transforms(config):
    '''Get train and validation augmentations.'''

    pre_transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                     else getattr(A, item['name'])(**item['params']) for item in config.augmentations.pre_transforms]
    transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                     else getattr(A, item['name'])(**item['params']) for item in config.augmentations.transforms]
    post_transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                      else getattr(A, item['name'])(**item['params']) for item in config.augmentations.post_transforms]

    train_transforms = A.Compose(pre_transforms + transforms + post_transforms)
    valid_transforms = A.Compose(pre_transforms + post_transforms)

    return train_transforms, valid_transforms


def get_train_file_path(image_id):
    return "/home/toefl/K/GLR/landmark-recognition-2021/train/{}/{}/{}/{}.jpg".format(image_id[0], image_id[1], image_id[2], image_id)

def data_generator(config):
    '''Generate data for train and validation splits.'''

    print('Getting the data')

    assert abs(config.data.train_size + config.data.val_size + config.data.test_size - 1.0) < 1e-9, \
                'sum of the sizes of splits must be equal to 1.0'

    data = pd.read_csv(config.paths.path_to_csv)
    data["id"] = data["id"].apply(get_train_file_path)
    m = {}
    for i, name in enumerate(data["landmark_id"].unique()):
        m[name] = i
    data["landmark_id"] = data["landmark_id"].map(m)


    if config.training.debug:
        data = data.sample(n=config.training.number_of_debug_samples, random_state=config.general.seed).reset_index(drop=True)

    if config.data.kfold.use_kfold:
        kfold = getattr(model_selection, config.data.kfold.name)(**config.data.kfold.params)
        
        if config.data.kfold.group_column:
            groups = data[config.data.kfold.group_column]
        else:
            groups = None
        
        for fold, (train_index, val_index) in enumerate(kfold.split(data, data[config.data.target_columns], groups)):
            if fold == config.data.kfold.current_fold:
                train_images = data[config.data.id_column].iloc[train_index].values
                train_targets = data[config.data.target_columns].iloc[train_index].values
                val_images = data[config.data.id_column].iloc[val_index].values
                val_targets = data[config.data.target_columns].iloc[val_index].values

                break
        
        if config.data.test_size == 0.0:
            return train_images, train_targets, val_images, val_targets
        
        val_size = config.data.val_size / (config.data.val_size + config.data.test_size)
        test_size = config.data.test_size / (config.data.val_size + config.data.test_size)
        val_images, test_images, val_targets, test_targets = train_test_split(val_images, val_targets,
                                                                              train_size=val_size,
                                                                              test_size=test_size,
                                                                              random_state=config.general.seed,
                                                                              stratify=val_targets)

        return train_images, train_targets, val_images, val_targets, test_images, test_targets

    else:
        train, val, _, _ = train_test_split(data,
                                            data[config.data.target_columns],
                                            train_size=config.data.train_size,
                                            test_size=config.data.val_size,
                                            random_state=config.general.seed,
                                            stratify=data[config.data.target_columns])

        train = train.drop_duplicates()
        val = val.drop_duplicates()

        train_images = train[config.data.id_column].values
        train_targets = train[config.data.target_columns].values
        val_images = val[config.data.id_column].values
        val_targets = val[config.data.target_columns].values

        return train_images, train_targets, val_images, val_targets


def get_loaders(config):
    '''Get data loaders.'''

    train_transforms, val_transorms = get_transforms(config)
    df = pd.read_csv(config.paths.path_to_csv)
    df["id"] = df["id"].apply(get_train_file_path)
    m = {}
    for i, name in enumerate(df["landmark_id"].unique()):
        m[name] = i
    df["landmark_id"] = df["landmark_id"].map(m)

    if config.training.debug:
        df = df.sample(n=config.training.number_of_debug_samples, random_state=config.general.seed).reset_index(drop=True)

    if config.data.test_size == 0.0:
        train_images, _, val_images, _ = data_generator(config)

        train_dataset = ClassificationDataset(config, df[df[config.data.id_column].isin(train_images)], train_transforms)
        val_dataset = ClassificationDataset(config, df[df[config.data.id_column].isin(val_images)], val_transorms)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.data.train_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)

        return train_loader, val_loader
    else:
        train_images, _, val_images, _, test_images, _ = data_generator(config)

        train_dataset = ClassificationDataset(config, df[df[config.data.id_column].isin(train_images)], train_transforms)
        val_dataset = ClassificationDataset(config, df[df[config.data.id_column].isin(val_images)], val_transorms)
        test_dataset = ClassificationDataset(config, df[df[config.data.id_column].isin(test_images)], val_transorms)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.data.train_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader