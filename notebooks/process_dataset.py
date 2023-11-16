""" Filter and tile dataset, save compressed. """
import sys
# remove last element from path
import os
from os.path import abspath, join, exists
import numpy as np
sys.path.pop()  # preexisting imports path messing up imports
sys.path.append(abspath(join('..')))  # ,'src'
print("\n".join(sys.path))
from src.data.dataset import CloudRawDataset, CloudProcessedDataset
import argparse

if __name__ == '__main__':
    # %% Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='/mnt/sdb1/code/sentinel2/interim', type=str)
    parser.add_argument('--output_dir', '-o',default='/mnt/sdb1/code/sentinel2/processed', type=str)
    args = parser.parse_args()
    # %% 
    dataset_kwargs = {'tile_size': 224, 'crop_pad_mask': 'crop'}
    path_filelist_train = join(args.input_dir, 'filelists', 'train.csv')
    path_filelist_val = join(args.input_dir, 'filelists', 'val.csv')
    path_filelist_test = join(args.input_dir, 'filelists', 'test.csv')
    # %% Load, process, save - validation set
    dataset_val = CloudRawDataset(path_filelist_val, args.input_dir, **dataset_kwargs)
    path_out_val = join(args.output_dir, 'val.npz')
    dataset_val.save_processed(path_out_val)
    # %% Load back dataset, check correctness
    path_out_val = join(args.output_dir, 'val.npz')
    dataset_val2 = CloudProcessedDataset(path_out_val)
    assert np.allclose(dataset_val[0]['image'], dataset_val2[0]['image'])
    assert np.allclose(dataset_val[0]['label'], dataset_val2[0]['label'])
    # %% test set
    # del dataset_val, dataset_val2
    path_out_test = join(args.output_dir, 'test.npz')
    dataset_test = CloudRawDataset(path_filelist_test, args.input_dir, **dataset_kwargs)
    dataset_test.save_processed(path_out_test)
    # %% train set
    # del dataset_test
    path_out_train = join(args.output_dir, 'train.npz')
    dataset_train = CloudRawDataset(path_filelist_train, args.input_dir, **dataset_kwargs)
    dataset_train.save_processed(path_out_train)