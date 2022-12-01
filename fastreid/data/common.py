# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, extra_transforms=None, relabel=True, sorted_id=True):
        self.img_items = img_items
        self.transform = transform
        self.extra_transforms = extra_transforms
        self.relabel = relabel

        if not sorted_id:
            self.pids = []
            self.cams = []
            for i in img_items:
                if i[1] not in self.pids:
                    self.pids.append(i[1])
                if i[2] not in self.cams:
                    self.cams.append(i[2])
        else:
            pid_set = set()
            cam_set = set()
            for i in img_items:
                pid_set.add(i[1])
                cam_set.add(i[2])
            self.pids = sorted(list(pid_set))
            self.cams = sorted(list(cam_set))

        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        domainid = img_item[3] if len(img_item) >= 4 and isinstance(img_item[3], int) else 0
        img = read_image(img_path)
        if self.extra_transforms is not None:
            img = self.extra_transforms(img)
        elif self.transform is not None:
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "domainids": domainid,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
