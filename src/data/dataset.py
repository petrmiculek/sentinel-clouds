""" Dataset class for clouds segmentation. """
# standard library
import os
from os.path import join

# external
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# local
from src.data.tiling import crop, pad, pad_with_mask, tile

class CloudDataset(Dataset):
    def __init__(self, path_filelist, path_data, tile_size=None, crop_pad_mask='crop'):
        self.path = path_data
        try:
            self.images_filenames = pd.read_csv(path_filelist, header=None)[0].values
        except Exception as e:
            print(e)
            self.images_filenames = np.array([])
        # pre-load images and labels
        self.images = []
        self.labels = []
        for f in self.images_filenames:
            x = np.load(join(path_data, 'subscenes', f))
            y = np.load(join(path_data, 'masks', f))
            x = x.transpose(2, 0, 1)  # HWC to CHW
            y = y.transpose(2, 0, 1)
            x = np.concatenate([x[:3], x[7:8]], axis=0)  # keep RGB + NIR channels (first 3 + 8th)
            # keep only labels cloudy and not cloudy
            y = y[1:2].astype(np.float32)  # cloudiness
            # y = np.concatenate([y[1:2], 1 - y[1:2]], axis=0)  # alternatively, if I decide for 2-class output (softmax)
            self.images.append(x)
            self.labels.append(y)
        # print(f'Dataset: {len(self.images)} samples')

        if tile_size is not None:
            self.crop_pad_mask = crop if crop_pad_mask == 'crop' else pad if crop_pad_mask == 'pad' else pad_with_mask
            self.tile_data(tile_size)
            # print(f'Dataset: {len(self.images)} tiles')

    def tile_data(self, tile_size):
        """ Cut images into tiles. 
        
        (C, N*H, N*W) -> (N*N, C, H, W)
        - keeping only full tiles
        - no overlap
        - assume square inputs of same shape, larger than tile_size
        - intermediate channel transpose back and forth (temporarily reverts work from constructor)
        """
        images = []
        labels = []
        def process(x):
            x = crop(x, tile_size)  # crop/pad/pad_with_mask
            x = tile(x, tile_size)
            return x

        for x, y in zip(self.images, self.labels):
            images.extend(process(x))
            labels.extend(process(y))

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, index):
        x = self.images[index]
        if x.shape[0] == 5:
            pad_mask = x[4:5]
            x = x[:4]
        else: 
            pad_mask = np.ones((1, x.shape[1], x.shape[2]), dtype=np.float32)
        y = self.labels[index]
        return {'image': x, 'label': y, 'pad_mask': pad_mask}   


def get_loaders(path_data, tile_size=None, crop_pad_mask='crop', **loader_kwargs):
    """ Dataset-loading wrapper - get train/val/test DataLoaders. """
    # TODO seed for workers
    path_filelist_train = join(path_data, 'filelists', 'train.csv')
    path_filelist_val = join(path_data, 'filelists', 'val.csv')
    path_filelist_test = join(path_data, 'filelists', 'test.csv')
    dataset_kwargs = {'path_data': path_data, 'tile_size': tile_size, 'crop_pad_mask': crop_pad_mask}
    dataset_train = CloudDataset(path_filelist_train, **dataset_kwargs)
    dataset_val = CloudDataset(path_filelist_val, **dataset_kwargs)
    dataset_test = CloudDataset(path_filelist_test, **dataset_kwargs)
    loader_train = DataLoader(dataset_train, shuffle=True, **loader_kwargs)
    loader_valid = DataLoader(dataset_val, **loader_kwargs)
    loader_test = DataLoader(dataset_test, **loader_kwargs)
    return {'train': loader_train, 'val': loader_valid, 'test': loader_test}
