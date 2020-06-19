# python -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --use_env train.py \
#     --coco_path /data/coco  \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
#     --output_dir /checkpoints/haianh/detr/mmcoco/ \
#     --epochs 20 \
#     --lr_drop 10



python  train.py \
    --coco_path /data/coco  \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output_dir /checkpoints/haianh/detr/mmcoco/ \
    --epochs 20 \
    --lr_drop 10