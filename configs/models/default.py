import mmcv
lr = 0.0001
lr_backbone = 1e-05
weight_decay = 0.0001
epochs = 20
lr_drop = 10
clip_max_norm = 0.1
frozen_weights = None
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1
nheads = 8
num_queries = 100
pre_norm = False
masks = False
aux_loss = True
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2

eos_coef = 0.1
dataset_file = 'ytvos'
coco_path = '/data/coco'
coco_panoptic_path = None
remove_difficult = False
device = 'cuda'
seed = 42
resume = 'https=//dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
start_epoch = 0
eval = False
num_workers = 2
world_size = 1
# dist_url= 'env=/
num_classes = 41
is_mm_model = True
output_dir = '/checkpoints/haianh/detr/ytvos/'
tb_logdir = '/checkpoints/haianh/detr/ytvos/tensorboard/'

weight_dict = {
    'loss_ce': 1,
    'loss_bbox': 5,
    'loss_object_matching': 1,
    'loss_giou': 2,
    # loss mass
    'loss_mask': 1,
    'loss_dice': 1,
}

dist_params = dict(backend='nccl')
launcher = "pytorch"
# launcher = 'none'

data = mmcv.Config.fromfile('configs/datasets/ytvos_tracking.py').data
checkpoint_freq = 1
gpu_ids = range(2)

debug=True
if debug:
    launcher = "none"
    data['train']['num_images'] = 10
    ds_train = data['train']
    data['train'] = dict(
        type='RepeatDataset',
        times=100,
        dataset=ds_train,
    )

    data['test'] = ds_train
    data['test']['flip_ratio'] = 0
    output_dir += '/debug'
    tb_logdir += '/debug'
