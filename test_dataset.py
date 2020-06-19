# from datasets import build_lm_wider_face


# dataset = build_lm_wider_face('train')
# item = dataset.__getitem__(0)

if __name__ == '__main__':
    import datasets
    import mmcv
    from mmdet.datasets import build_dataset
    cfg = mmcv.Config.fromfile('./configs/datasets/ytvos_tracking.py')
    dataset = build_dataset(cfg.data['train'])
    item = dataset.__getitem__(0)
    import pdb; pdb.set_trace()