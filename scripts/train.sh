export DEBUG=0
python main.py --batch_size 8 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_dir 'work_dirs/'

# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#     --batch_size 8 \
#     --resume work_dirs/voc/checkpoint.pth