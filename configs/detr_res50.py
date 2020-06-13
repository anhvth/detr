lr = 0.0001
lr_backbone = 1e-05
batch_size = 2
weight_decay = 0.0001
epochs = 30
lr_drop = 20
clip_max_norm = 0.1
frozen_weights = None
backbone = "resnet50"
dilation = False
position_embedding = "sine"
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1
nheads = 8
num_queries = 100
pre_norm = False
masks = False
aux_loss = False
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2
mask_loss_coef = 1
dice_loss_coef = 1
bbox_loss_coef = 5
giou_loss_coef = 2
eos_coef = 0.1
dataset_file = "coco"
coco_path = "/data/coco"
coco_panoptic_path = None
remove_difficult = False
output_dir = ""
device = "cuda"
seed = 42
resume = "./work_dirs/checkpoint.pth"
start_epoch = 0
eval = True
num_workers = 2
world_size = 1
dist_url = "env://"
# parser.add_argument('--bbox_loss_coef', default=5, type=float)

losses = ['labels', 'boxes', 'cardinality', 'arcface']
weight_dict={'loss_ce': 1, 'loss_bbox':5, 'loss_angular':3}
sampler_num = 1000*100
dataset_type = 'voc'
coco_path = '/data/coco/'
remove_difficult=False

num_classes = 20 if dataset_type == 'voc' else 91

work_dir='work_dirs/voc/'
