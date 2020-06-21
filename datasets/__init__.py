# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .ytvos import build as build_ytvos
from .lm_widerface import build as build_lm_wider_face
# from .ytvos import YTVOSDataset, DETRWraper_YTVOSDataset
from .extra_aug import ExtraAugmentation
# from .pipelines import 
import mmcv
from pyson import timeit
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

@timeit
def build_dataset(image_set, cfg):
    if cfg.dataset_file == 'coco':
        return build_coco(image_set, cfg)
    if cfg.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, cfg)
    if cfg.dataset_file == 'ytvos':
        print(f'Building {image_set}\n {cfg.data}')
        return build_ytvos(image_set, cfg.data)
    
    raise ValueError(f'dataset {cfg.dataset_file} not supported')
