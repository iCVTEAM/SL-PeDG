# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import random
import torch

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from fastreid.utils.file_io import PathManager


@DATASET_REGISTRY.register()
class DG_Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ""
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        if 'domain' in kwargs.keys():
            self.domain_id = kwargs['domain']
        else:
            self.domain_id = 0

        cfg = kwargs['cfg']
        is_val = cfg.DATASETS.SPLIT_VAL
        val_pids = cfg.DATASETS.VAL_PIDS

        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.val_pid_list = []
        self.train_pid_list = []

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)

        query, val_query = self.split_val(self.query_dir, is_train=True, is_val=is_val, val_pids=val_pids)
        gallery, val_gallery = self.split_val(self.gallery_dir, is_train=True, is_val=is_val, val_pids=val_pids)

        train = train + query + gallery

        super(DG_Market1501, self).__init__(train, val_query, val_gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored

            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            if pid not in self.train_pid_list:
                self.train_pid_list.append(pid)
            data.append((img_path, pid, camid, self.domain_id))

        return data

    def split_val(self, dir_path, is_train=True, is_val=False, val_pids=50):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        val_data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored

            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                if is_val and (len(self.val_pid_list) < val_pids or pid in self.val_pid_list):
                    val_data.append((img_path, pid, camid, self.domain_id))
                    if pid not in self.val_pid_list: self.val_pid_list.append(pid)

                    if self.domain_centers is not None:
                        val_data[-1] = val_data[-1] + (self.domain_centers,)
                else:
                    pid = self.dataset_name + "_" + str(pid)
                    camid = self.dataset_name + "_" + str(camid)
                    data.append((img_path, pid, camid, self.domain_id))
                    if pid != self.dataset_name + "_0" and pid not in self.train_pid_list:
                        self.train_pid_list.append(pid)

            else:
                val_data.append((img_path, pid, camid, self.domain_id))

        return data, val_data
