# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import json
import os
import random
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
# import argparse
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from mmdet.datasets import build_dataloader
from models import build_model
from pyson.utils import memoize

# os.environ['LOCAL_RANK'] = str(0)
# os.environ['RANK'] = str(0)
# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector',
#                                      add_help=False)
#     parser.add_argument('--lr', default=1e-4, type=float)
#     parser.add_argument('--lr_backbone', default=1e-5, type=float)
#     parser.add_argument('--batch_size', default=2, type=int)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--lr_drop', default=200, type=int)
#     parser.add_argument('--clip_max_norm',
#                         default=0.1,
#                         type=float,
#                         help='gradient clipping max norm')
#     # Model parameters
#     parser.add_argument(
#         '--frozen_weights',
#         type=str,
#         default=None,
#         help=
#         "Path to the pretrained model. If set, only the mask head will be trained"
#     )
#     # * Backbone
#     parser.add_argument('--backbone',
#                         default='resnet50',
#                         type=str,
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument(
#         '--dilation',
#         action='store_true',
#         help=
#         "If true, we replace stride with dilation in the last convolutional block (DC5)"
#     )
#     parser.add_argument(
#         '--position_embedding',
#         default='sine',
#         type=str,
#         choices=('sine', 'learned'),
#         help="Type of positional embedding to use on top of the image features"
#     )

#     # * Transformer
#     parser.add_argument('--enc_layers',
#                         default=6,
#                         type=int,
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers',
#                         default=6,
#                         type=int,
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument(
#         '--dim_feedforward',
#         default=2048,
#         type=int,
#         help=
#         "Intermediate size of the feedforward layers in the transformer blocks"
#     )
#     parser.add_argument(
#         '--hidden_dim',
#         default=256,
#         type=int,
#         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout',
#                         default=0.1,
#                         type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument(
#         '--nheads',
#         default=8,
#         type=int,
#         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_queries',
#                         default=100,
#                         type=int,
#                         help="Number of query slots")
#     parser.add_argument('--pre_norm', action='store_true')

#     # * Segmentation
#     parser.add_argument('--masks',
#                         action='store_true',
#                         help="Train segmentation head if the flag is provided")

#     # Loss
#     parser.add_argument(
#         '--no_aux_loss',
#         dest='aux_loss',
#         action='store_false',
#         help="Disables auxiliary decoding losses (loss at each layer)")
#     # * Matcher
#     parser.add_argument('--set_cost_class',
#                         default=1,
#                         type=float,
#                         help="Class coefficient in the matching cost")
#     parser.add_argument('--set_cost_bbox',
#                         default=5,
#                         type=float,
#                         help="L1 box coefficient in the matching cost")
#     parser.add_argument('--set_cost_giou',
#                         default=2,
#                         type=float,
#                         help="giou box coefficient in the matching cost")
#     # * Loss coefficients
#     parser.add_argument('--mask_loss_coef', default=1, type=float)
#     parser.add_argument('--dice_loss_coef', default=1, type=float)
#     parser.add_argument('--bbox_loss_coef', default=5, type=float)
#     parser.add_argument('--giou_loss_coef', default=2, type=float)
#     parser.add_argument(
#         '--eos_coef',
#         default=0.1,
#         type=float,
#         help="Relative classification weight of the no-object class")

#     # dataset parameters
#     parser.add_argument('--dataset_file', default='coco')
#     parser.add_argument('--coco_path', type=str)
#     parser.add_argument('--coco_panoptic_path', type=str)
#     parser.add_argument('--remove_difficult', action='store_true')

#     parser.add_argument('--output_dir',
#                         default='',
#                         help='path where to save, empty for no saving')
#     parser.add_argument('--device',
#                         default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--seed', default=42, type=int)
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--start_epoch',
#                         default=0,
#                         type=int,
#                         metavar='N',
#                         help='start epoch')
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--num_workers', default=2, type=int)

#     # distributed training parameters
#     parser.add_argument('--world_size',
#                         default=1,
#                         type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--dist_url',
#                         default='env://',
#                         help='url used to set up distributed training')
#     return parser


def main(args):

    # utils.init_distributed_mode(args)
    # import ipdb; ipdb.set_trace()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **args.dist_params)

    # print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        from mmcv.runner import load_state_dict
        # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        load_state_dict(model_without_ddp, checkpoint['model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            load_state_dict(lr_scheduler, checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # model = MMDataParallel(model)

    # put model on gpus

    if distributed:
        find_unused_parameters = args.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    else:
        model = MMDataParallel(model.cuda(args.gpu_ids[0]),
                               device_ids=args.gpu_ids)
    rank, _ = get_dist_info()

    dataset_train = build_dataset(image_set='train', args=args)
    data_loader_train = build_dataloader(dataset_train,
                                         args.data['imgs_per_gpu'],
                                         args.data['workers_per_gpu'],
                                         num_gpus=torch.cuda.device_count(),
                                         dist=distributed,
                                         seed=1)

    args.tb_logdir = f'{args.tb_logdir}/{rank}'
    writer = SummaryWriter(args.tb_logdir)
    os.system(f'rm -r {args.tb_logdir}')
    os.makedirs(args.tb_logdir, exist_ok=True)
    print(f'Tensorboard: tensorboard --logdir {args.tb_logdir}')
    # print(rank, type(rank))
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model,
                                      criterion,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      args.clip_max_norm,
                                      writer=writer)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (
                    epoch + 1) % args.checkpoint_freq == 0:
                checkpoint_paths.append(output_dir /
                                        f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        # 'model': model.module.state_dict(),
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                    },
                    checkpoint_path)
                

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = mmcv.Config.fromfile('./configs/models/default.py')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
