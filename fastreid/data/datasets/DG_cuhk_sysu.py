# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import random
import torch

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from fastreid.utils.file_io import PathManager


@DATASET_REGISTRY.register()
class DG_CUHK_SYSU(ImageDataset):
    """CUHK SYSU datasets.

    The dataset is collected from two sources: street snap and movie.
    In street snap, 12,490 images and 6,057 query persons were collected
    with movable cameras across hundreds of scenes while 5,694 images and
    2,375 query persons were selected from movies and TV dramas.

    Dataset statistics:
        - identities: xxx.
        - images: 12936 (train).
    """
    dataset_dir = "CUHK-SYSU"
    dataset_name = "cuhksysu"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.domain_id = kwargs['domain']
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_pid_list = []

        self.data_dir = osp.join(self.dataset_dir, "cropped_images")

        required_files = [self.data_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.data_dir)
        query = []
        gallery = []

        super(DG_CUHK_SYSU, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'p([-\d]+)_n(\d)')

        data = []
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_0"
            if pid not in self.train_pid_list:
                self.train_pid_list.append(pid)
            data.append((img_path, pid, camid, self.domain_id))

        return data
