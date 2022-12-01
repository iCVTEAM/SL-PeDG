# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.utils.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    def __init__(self, cfg):
        super().__init__()

        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        scale = cfg.MODEL.HEADS.SCALE
        margin = cfg.MODEL.HEADS.MARGIN
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim))

        self.bottleneck = nn.Sequential(*neck)
        self.bottleneck.apply(weights_init_kaiming)

        # Linear layer
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        # Initialize weight parameters
        if cls_type == "Linear":
            nn.init.normal_(self.weight, std=0.001)
        elif cls_type == "CircleSoftmax":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif cls_type == "ArcSoftmax" or cls_type == "CosSoftmax":
            nn.init.xavier_uniform_(self.weigth)

        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

    def forward(self, features, feature_list, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        # if self.cls_layer.__class__.__name__ == 'Linear':
        #     logits = F.linear(neck_feat, self.weight)
        # else:
        #     logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        logits = F.linear(neck_feat, self.weight)

        cls_outputs = self.cls_layer(logits, targets)

        # fmt: off
        if self.neck_feat == 'before':
            feat = pool_feat[..., 0, 0]
        else:
            feat = neck_feat
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits * self.cls_layer.s,
            "features": feat,
        }