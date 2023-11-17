""" Dataset class for clouds segmentation. """
# standard library
import os
from os.path import join, dirname, dirname

# external
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm

# local
from src.data.tiling import crop, pad, pad_with_mask, tile

class CloudRawDataset(Dataset):
    def __init__(self, path_filelist, path_data, tile_size=None, crop_pad_mask='crop'):
        self.path_filelist = path_filelist
        self.path_filelist = path_filelist
        self.path = path_data
        self.tile_size = tile_size
        self.crop_pad_mask = crop_pad_mask  # TODO unused
        try:
            self.images_filenames = pd.read_csv(path_filelist, header=None)[0].values
        except Exception as e:
            print(e)
            self.images_filenames = np.array([])
        # pre-load images and labels
        self.images = []
        self.labels = []
        for f in tqdm(self.images_filenames, desc=f'Loading: {path_filelist}'):
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

    # multiprocessing - unused
    """
    from multiprocessing import Pool, cpu_count
    with Pool() as p:
    self.images = p.map(self.load_x, self.images_filenames)
    self.labels = p.map(self.load_y, self.images_filenames)
    """
    def load_x(self, f):
        x = np.load(join(self.path, 'subscenes', f))
        x = x.transpose(2, 0, 1)  # HWC to CHW
        x = np.concatenate([x[:3], x[7:8]], axis=0)  # keep RGB + NIR channels (first 3 + 8th)
        return x

    def load_y(self, f):
        y = np.load(join(self.path, 'masks', f))
        y = y.transpose(2, 0, 1)
        y = y[1:2].astype(np.float32)  # cloudiness
        return y

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

        for x, y in tqdm(zip(self.images, self.labels), total=len(self.images), desc='Tiling data'):
            images.extend(process(x))
            labels.extend(process(y))

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        if x.shape[0] == 5:
            pad_mask = x[4:5]
            x = x[:4]
        else: 
            pad_mask = np.ones((1, x.shape[1], x.shape[2]), dtype=np.float32)
        y = self.labels[index]
        # label smoothing  TODO parameter
        y = y * 0.9 + 0.05
        return {'image': x, 'label': y, 'pad_mask': pad_mask}
    
    def save_processed(self, path_save):
        """ Save processed dataset to disk. """
        os.makedirs(dirname(path_save), exist_ok=True)
        os.makedirs(dirname(path_save), exist_ok=True)
        np.savez(path_save, images=self.images, labels=self.labels)
    
class CloudProcessedDataset(CloudRawDataset):
    def __init__(self, path_data, **kwargs):
        if len(kwargs) > 0:
            print(f'Ignoring dataset args: {kwargs}')
        # load npz with images and labels
        data = np.load(path_data)
        self.images = data['images']
        self.labels = data['labels']
        self.tile_size = None  # TODO unused, can't tell that from npz
        self.crop_pad_mask = 'crop'
        if self.images[0].shape[0] == 5:
            self.crop_pad_mask = 'pad_with_mask'

        # TODO debug
        # limit = 4
        # step_size = 21
        # self.images = self.images[:limit * step_size:step_size]
        # self.labels = self.labels[:limit * step_size:step_size]

def get_loaders(dir_data, tile_size=None, crop_pad_mask='crop', **loader_kwargs):
    """ Dataset-loading wrapper - get train/val/test DataLoaders. """
    # TODO seed for workers
    shuffle_train = loader_kwargs.pop('shuffle', True)
    path_filelist_train = join(dir_data, 'filelists', 'train.csv')
    path_filelist_val = join(dir_data, 'filelists', 'val.csv')
    path_filelist_test = join(dir_data, 'filelists', 'test.csv')
    dataset_kwargs = {'path_data': dir_data, 'tile_size': tile_size, 'crop_pad_mask': crop_pad_mask}
    dataset_train = CloudRawDataset(path_filelist_train, **dataset_kwargs)
    dataset_val = CloudRawDataset(path_filelist_val, **dataset_kwargs)
    dataset_test = CloudRawDataset(path_filelist_test, **dataset_kwargs)
    loader_train = DataLoader(dataset_train, shuffle=shuffle_train, **loader_kwargs)
    loader_valid = DataLoader(dataset_val, **loader_kwargs)
    loader_test = DataLoader(dataset_test, **loader_kwargs)
    return {'train': loader_train, 'val': loader_valid, 'test': loader_test}

def get_loaders_processed(dir_data_processed, splits=None, **loader_kwargs):
    """ Dataset-loading wrapper - get train/val/test DataLoaders. """
    if splits is None:  # get all splits
        splits = ['train', 'val', 'test']
    # TODO seed for workers
    tile_size = loader_kwargs.pop('tile_size', None)
    crop_pad_mask = loader_kwargs.pop('crop_pad_mask', None)
    if tile_size is not None or crop_pad_mask is not None:
        print(f'Ignoring dataset args: {loader_kwargs}, loading dataset already processed.')
    shuffle_train = loader_kwargs.pop('shuffle', True)
    loaders = dict()
    for split in splits:
        assert split in ['train', 'val', 'test']
        shuffle = shuffle_train if split == 'train' else True  # TODO changed for debug
        dataset = CloudProcessedDataset(join(dir_data_processed, f'{split}.npz'))
        loader = DataLoader(dataset, shuffle=shuffle, **loader_kwargs)
        loaders[split] = loader
    return loaders