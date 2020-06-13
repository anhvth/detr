work_dir  = '../checkpoints/voc/atss_r50_voc15/'
data_root = '/data/VOC/VOCdevkit/'
load_from = None
resume_from = work_dir + 'latest.pth'

repeat_times = 3
lr_start = 5e-3
lr_end = 5e-5
optimizer = dict(type='SGD', lr=lr_start, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
train_ann_file=[
	data_root + 'VOC2007/ImageSets/Main/trainval.txt',
	data_root + 'VOC2012/ImageSets/Main/trainval.txt'
]
train_img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/']

imgs_per_gpu = 8
workers_per_gpu = 6
total_epochs = 12
lr_config = dict(
	policy='cosine', target_lr=lr_end, by_epoch=False,
	warmup='linear', warmup_iters=500, warmup_ratio=1.0/3,
)
checkpoint_config = dict(interval=1)
log_config = dict(
	interval=50,
	hooks=[
		dict(type='TextLoggerHook'),
	]
)
num_used_classes = None
ret_logits = True

# dataset settings
dataset_type = 'VOCDataset'
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53],
	std=[58.395, 57.12, 57.375],
	to_rgb=True,
)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
# 	dict(type='Resize', img_scale=[(600, 600), (1000, 1000)], keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.0),
# 	dict(type='Normalize', **img_norm_cfg),
# 	dict(type='Pad', size_divisor=32),
# 	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(1000, 600),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']),
		])
]
data = dict(
	imgs_per_gpu=imgs_per_gpu,
	workers_per_gpu=workers_per_gpu,
	train=dict(
		type='RepeatDataset',
		times=repeat_times,
		dataset=dict(
			type=dataset_type,
			ann_file=train_ann_file,
			img_prefix=train_img_prefix,
			pipeline=train_pipeline,
			num_used_classes=num_used_classes)),
	val=dict(
		type=dataset_type,
		ann_file=data_root + 'VOC2007TEST/ImageSets/Main/test.txt',
		img_prefix=data_root + 'VOC2007TEST/',
		pipeline=test_pipeline,
		num_used_classes=num_used_classes),
	test=dict(
		type=dataset_type,
		ann_file=data_root + 'VOC2007TEST/ImageSets/Main/test.txt',
		img_prefix=data_root + 'VOC2007TEST/',
		pipeline=test_pipeline,
		num_used_classes=num_used_classes,
		test_mode=True))

evaluation = dict(interval=1, metric='mAP')
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
device=range(8)
