from glob import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
# from cython_bbox import bbox_overlaps as bbox_ious
# from opts import opts
# from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
# from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta


def JointTrackingDataset(Dataset):
    def __init__(self, root):
        self.label_files = glob(f'{root}/**/*.txt', recursive=True)
        # for ds, path in paths.items():
        #     self.label_files[ds] = [
        #         x.replace('images', 'labels_with_ids').replace(
        #             '.png', '.txt').replace('.jpg', '.txt')
        #         for x in self.img_files[ds]]