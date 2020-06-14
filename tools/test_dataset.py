from mmdet.datasets import build_dataloader, build_dataset
import mmcv

if __name__ == '__main__':
    cfg = mmcv.Config.fromfile('configs/datasets/coco_detection.py')
    dataset = build_dataset(cfg.data.test)
    item = dataset.__getitem__(0)