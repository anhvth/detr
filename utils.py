import os
from mmcv import Config
debug = os.environ.get('DEBUG', '0') == '1'

cfg = Config.fromfile('train_cfg.py')