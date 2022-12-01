# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import re
import glob
import os.path as osp
import torch

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from fastreid.utils.file_io import PathManager


@DATASET_REGISTRY.register()
class DG_MSMT17(ImageDataset):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = "MSMT17"
    dataset_url = None
    dataset_name = 'MSMT17'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.domain_id = kwargs['domain']

        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=True)
        gallery = self.process_dir(self.gallery_dir, is_train=True)

        train = train + query + gallery

        super(DG_MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, self.domain_id))

        return data
