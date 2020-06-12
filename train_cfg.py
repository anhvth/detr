# parser.add_argument('--bbox_loss_coef', default=5, type=float)

losses = ['labels', 'boxes', 'cardinality', 'arcface']
weight_dict={'loss_ce': 1, 'loss_bbox':5, 'loss_angular':3}
sampler_num = 1000*100
dataset_type = 'voc'
coco_path = '/data/coco/'
remove_difficult=False

num_classes = 20 if dataset_type == 'voc' else 91

work_dir='work_dirs/voc/'