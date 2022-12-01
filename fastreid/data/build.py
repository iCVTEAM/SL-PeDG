# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import os
import numpy as np

import torch
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader

from fastreid.config import configurable
from fastreid.utils import comm
from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms, build_extra_transforms

__all__ = [
    "build_reid_train_loader",
    "build_reid_val_loader",
    "build_reid_test_loader"
]


def _train_loader_from_config(cfg, *, Dataset=None, transforms=None, sampler=None, **kwargs):
    if transforms is None:
        transforms = build_transforms(cfg, is_train=True)

    if Dataset is None:
        Dataset = CommDataset

    train_items = list()
    for idx, d in enumerate(cfg.DATASETS.NAMES):
        data = DATASET_REGISTRY.get(d)(root=cfg.DATASETS.ROOT, domain=idx, cfg=cfg, **kwargs)
        if comm.is_main_process():
            data.show_train()
        train_items.extend(data.train)

    train_set = Dataset(train_items, transforms, relabel=True, sorted_id=cfg.DATASETS.SORTED_ID)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        num_instance = cfg.DATALOADER.NUM_INSTANCE
        domain_shuffle = cfg.DATALOADER.DOMAIN_SHUFFLE
        domain_step = cfg.DATALOADER.DOMAIN_STEP
        step_iters = cfg.DATALOADER.STEP_ITERS
        num_src_domains = cfg.DATALOADER.NUM_SRC_DOMAINS
        min_subset_len = cfg.DATALOADER.MIN_SUBSET_LEN
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(train_set))
        elif sampler_name == "NaiveIdentitySampler":
            sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
        elif sampler_name == "BalancedIdentitySampler":
            sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size,
                                                       num_instance)
        elif sampler_name == "DomainSplitBalancedSampler":
            sampler = samplers.DomainSplitBalancedSampler(train_set.img_items, mini_batch_size, num_instance,
                                                                domain_step, domain_shuffle, num_src_domains,
                                                                min_subset_len, step_iters)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "train_set": train_set,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_reid_train_loader(
        train_set, *, sampler=None, total_batch_size, num_workers=0,
):
    """
    Build a dataloader for object re-identification with some default features.
    This interface is experimental.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """
    if isinstance(sampler, list):
        mini_meta_train_batch_size = total_batch_size[0] // comm.get_world_size()
        mini_meta_test_batch_size = total_batch_size[1] // comm.get_world_size()

        meta_train_batch_sampler = torch.utils.data.sampler.BatchSampler(sampler[0], mini_meta_train_batch_size, True)
        meta_test_batch_sampler = torch.utils.data.sampler.BatchSampler(sampler[1], mini_meta_test_batch_size, True)

        meta_train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=num_workers,
            batch_sampler=meta_train_batch_sampler,
            collate_fn=fast_batch_collator,
            pin_memory=True,
        )
        meta_test_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=num_workers,
            batch_sampler=meta_test_batch_sampler,
            collate_fn=fast_batch_collator,
            pin_memory=True,
        )
        return [meta_train_loader, meta_test_loader]

    mini_batch_size = total_batch_size // comm.get_world_size()

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def _test_loader_from_config(cfg, dataset_name, *, Dataset=None, transforms=None, **kwargs):
    if transforms is None:
        transforms = build_transforms(cfg, is_train=False)

    if Dataset is None:
        Dataset = CommDataset

    data = DATASET_REGISTRY.get(dataset_name)(root=cfg.DATASETS.ROOT, cfg=cfg, **kwargs)
    if comm.is_main_process():
        data.show_test()
    test_items = data.query + data.gallery

    test_set = Dataset(test_items, transforms, relabel=False)

    return {
        "test_set": test_set,
        "test_batch_size": cfg.TEST.IMS_PER_BATCH,
        "num_query": len(data.query),
    }


@configurable(from_config=_test_loader_from_config)
def build_reid_test_loader(test_set, test_batch_size, num_query, num_workers=4):
    """
    Similar to `build_reid_train_loader`. This sampler coordinates all workers to produce
    the exact set of all samples
    This interface is experimental.

    Args:
        test_set:
        test_batch_size:
        num_query:
        num_workers:

    Returns:
        DataLoader: a torch DataLoader, that loads the given reid dataset, with
        the test-time transformation.

    Examples:
    ::
        data_loader = build_reid_test_loader(test_set, test_batch_size, num_query)
        # or, instantiate with a CfgNode:
        data_loader = build_reid_test_loader(cfg, "my_test")
    """

    mini_batch_size = test_batch_size // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, num_query


def _val_loader_from_config(cfg, dataset_name, *, Dataset=None, transforms=None, **kwargs):
    if transforms is None:
        transforms = build_transforms(cfg, is_train=True)

    if Dataset is None:
        Dataset = CommDataset

    data = DATASET_REGISTRY.get(dataset_name)(root=cfg.DATASETS.ROOT, cfg=cfg, **kwargs)
    if comm.is_main_process():
        if kwargs['is_train']:
            data.show_train()
        else:
            data.show_test()
    test_items = data.train

    test_set = Dataset(test_items, transforms, relabel=False)

    return {
        "val_set": test_set,
        "val_batch_size": cfg.TEST.IMS_PER_BATCH,
    }


@configurable(from_config=_val_loader_from_config)
def build_reid_val_loader(val_set, val_batch_size, num_workers=4):
    """
    Similar to `build_reid_train_loader`. This sampler coordinates all workers to produce
    the exact set of all samples
    This interface is experimental.
    """
    mini_batch_size = val_batch_size // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(val_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    val_loader = DataLoader(
        val_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return val_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
