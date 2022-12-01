# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob
import torch

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from fastreid.utils.file_io import PathManager

__all__ = ['DG_VIPER', ]


@DATASET_REGISTRY.register()
class DG_VIPER(ImageDataset):
    dataset_dir = "DGTEST/viper"
    dataset_name = "viper"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        subset = kwargs['opt']

        self.train_dir = os.path.join(self.root, self.dataset_dir, subset, 'train')
        self.query_dir = os.path.join(self.root, self.dataset_dir, subset, 'query')
        self.gallery_dir = os.path.join(self.root, self.dataset_dir, subset, 'gallery')

        required_files = [
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_train(self.train_dir, is_train=True)
        query = self.process_train(self.query_dir, is_train=False)
        gallery = self.process_train(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, path, is_train=True):
        data = []
        img_list = glob(os.path.join(path, '*.png'))
        for img_path in img_list:
            img_name = img_path.split('/')[-1]  # p000_c1_d045.png
            split_name = img_name.split('_')
            pid = int(split_name[0][1:])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            camid = int(split_name[1][1:])
            # dirid = int(split_name[2][1:-4])
            data.append([img_path, pid, camid])

        return data
