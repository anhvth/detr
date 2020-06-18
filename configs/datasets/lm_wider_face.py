imgs_per_gpu = 16
size_divisor = 32
repeat_times = 1
workers_per_gpu = 6
keep_ratio = True
img_scale_test = (int(640 * 1.5), int(480 * 1.5))
img_scale = [(640, 480), img_scale_test]

dataset_type = 'CWiderFaceDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

data_root = '/data/widerface/'

train_ann_file = data_root + 'annotations/train.json'
val_ann_file = data_root + 'annotations/val.json'
test_ann_file = data_root + 'annotations/val.json'

train_img_prefix = data_root + 'WIDER_train'
val_img_prefix = data_root + 'WIDER_val'
test_img_prefix = data_root + 'WIDER_val'
min_size = 5


train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_landmark=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Expand',
         mean=img_norm_cfg['mean'],
         to_rgb=img_norm_cfg['to_rgb'],
         ratio_range=(1, 2),
         prob=0.5,
         ),
    dict(type='MinIoURandomCrop',
         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
         min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=keep_ratio),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=size_divisor),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=[
             'img', 'gt_bboxes', 'gt_labels', 'gt_landmarks',
             'gt_landmarks_weight'
         ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=size_divisor),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(type='RepeatDataset',
               times=repeat_times,
               dataset=dict(
                   type=dataset_type,
                   ann_file=train_ann_file,
                   img_prefix=train_img_prefix,
                   min_size=min_size,
                   pipeline=train_pipeline,
               )),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        img_prefix=test_img_prefix,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# if __name__ == '__main__':
#     import mmcv
#     from torch.utils.data import Dataset, DataLoader
#     from mmdet.datasets import build_dataset
#     import datasets

#     cfg = mmcv.Config(data)
#     dataset = build_dataset(cfg.val)
#     import pdb; pdb.set_trace()