# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Vehicle re-id datasets
from .DG_cuhk_sysu import DG_CUHK_SYSU
from .DG_cuhk02 import DG_CUHK02
from .DG_cuhk03_labeled import DG_CUHK03_labeled
from .DG_cuhk03_detected import DG_CUHK03_detected
from .DG_dukemtmcreid import DG_DukeMTMC
from .DG_market1501 import DG_Market1501
from .DG_msmt17 import DG_MSMT17

from .DG_prid import DG_PRID
from .DG_grid import DG_GRID
from .DG_iLIDS import DG_ILIDS
from .DG_viper import DG_VIPER


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
