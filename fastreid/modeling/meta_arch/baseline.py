# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    def __init__(self, cfg):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        self.heads = build_heads(cfg)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1), False)

        self.loss_kwargs = self.get_loss_kwargs(cfg)

    def get_loss_kwargs(self, cfg):
        return {
            # loss name
            'loss_names': cfg.MODEL.LOSSES.NAME,

            # loss hyperparameters
            'ce': {
                'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                'scale': cfg.MODEL.LOSSES.CE.SCALE
            },
            'tri': {
                'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                'scale': cfg.MODEL.LOSSES.TRI.SCALE
            },
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, suffix=''):
        images = self.preprocess_image(batched_inputs)
        features, features_list = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)
            cams = batched_inputs["camids"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, features_list, targets)
            losses = self.losses(outputs, targets, cams, suffix=suffix)
            return losses
        else:
            outputs = self.heads(features, features_list)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images'].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels, gt_cams=None, suffix=''):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls' + suffix] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet' + suffix] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        return loss_dict

