import json
import os
import os.path as osp
import sys

import mmcv
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
# from TrackTransMOT.src import _init_paths
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname('./TrackTransMOT/src/_init_paths.py')

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)




from datasets.dataset_factory import get_dataset
from logger import Logger
from opts import opts
from trains.train_factory import train_factory

opt = mmcv.Config.fromfile('TrackTransMOT/src/lib/cfg/default_opts.py')

Dataset = get_dataset(opt.dataset, opt.task)
print("Dataset:", Dataset)
if ".json" in opt.data_cfg:
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
elif ".py" in opt.data_cfg:
    data_config = mmcv.Config.fromfile(opt.data_cfg).data
    trainset_paths = data_config["train"]
    dataset_root = data_config["root"]
else:
    raise NotImplemented

transforms = T.Compose([T.ToTensor()])

dataset = Dataset(opt,
                  dataset_root,
                  trainset_paths, (1088, 608),
                  augment=True,
                  transforms=transforms)
