import mmcv
import matplotlib.pyplot as plt
import requests
import torch
import torchvision.transforms as T
from mmcv import Config
from PIL import Image
from torch import nn
from torchvision.models import resnet50
from tqdm import tqdm

from mmdet.datasets import build_dataloader, build_dataset
from models.angular_loss import AngularMarginLoss
from models.backbone import build_backbone
from models.detr import DETR
from models.transformer import Transformer, build_transformer
import numpy as np
from torch.nn import functional as F
torch.set_grad_enabled(False)

thr_prob = 0.3

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def tensor_to_image(img):
    img = img[0]
    img = img.permute([1,2,0])
    img = (img-img.min()) / (img.max()-img.min())
    img = img*255
    img = img.cpu().numpy().astype('uint8')
    return Image.fromarray(img[:,:,::-1])


CLASSES = ('N/A', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

# build model

args = Config.fromfile('./cfg.py')
cfg = Config.fromfile('./F6_config.py')
backbone = build_backbone(args)
transformer = build_transformer(args)


model = DETR(
    backbone,
    transformer,
    num_classes=19,
    num_queries=args.num_queries,
    aux_loss=args.aux_loss,
    ang_loss=AngularMarginLoss(256, num_classes=20)
)

ckpt = torch.load('./work_dirs/voc/checkpoint.pth')
model.load_state_dict(ckpt['model'], strict=True)

# build dataset

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    imgs_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
# Eval
pbar = tqdm(data_loader, total=len(data_loader))
#model = nn.DataParallel(model)
model.eval()
results = []
model.to('cuda')

for item in pbar:

    img = item['img'][0].cuda()

    meta = item['img_metas'][0].data[0][0]

    h, w, c = meta['ori_shape']
    with torch.no_grad():
        outputs = model(img)

    pred = outputs['pred_logits'].sigmoid()[0]
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
    keep = probas.max(-1).values > thr_prob
    size = (w, h)
    # take keep
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), size)
    if sum(keep) > 0:
        scores, classes = probas[keep].max(1)
        embed = outputs['arcface_input'][0][keep]
        embed = F.normalize(embed)

        
        boxes=bboxes_scaled.cpu().numpy()
        scores = scores.cpu().numpy()[:, None]
        boxes_with_scores = np.concatenate([boxes, scores], 1)
        result = dict(
            boxes= boxes_with_scores,
            classes=classes.cpu().numpy(),
            embed=embed.cpu().numpy(),
        )
    else:
        result = dict(
            boxes=np.zeros([0, 5], dtype=np.float32),
            classes=np.array([], dtype=np.int64),
            embed=np.array([], dtype=np.float32)
        )

    results.append(result)

mmcv.dump(results, 'out.pkl')
