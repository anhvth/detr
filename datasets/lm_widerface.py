import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from pycocotools.coco import COCO
from mmdet.datasets import DATASETS, CocoDataset, XMLDataset


@DATASETS.register_module
class CWiderFaceDataset(CocoDataset):
    """
    Reader for the WIDER Face dataset in  COCO format.
    Conversion scripts can be found in
    ccdetection/tools/convert_datasets/convert_widerface_to_coco.py
    """
    CLASSES = ('face', )

    def __init__(self, min_size=None, **kwargs):
        super(CWiderFaceDataset, self).__init__(**kwargs)
        item = self.__getitem__(0)

    def load_annotations(self, ann_file):
        import pdb; pdb.set_trace()
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_landmarks = []
        gt_landmarks_weight = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            landmark = ann['landmark']
            gt_bboxes.append(bbox)
            gt_landmarks.append(landmark)
            lm_weight = 0 if (np.array(landmark) < 0).any() else 1
            gt_landmarks_weight.append(lm_weight)
            gt_labels.append(self.cat2label[ann['category_id']])
            # gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_landmarks = np.array(gt_landmarks, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_landmarks_weight = np.array(
                gt_landmarks_weight, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_landmarks = np.zeros((0, 14), dtype=np.float32)
            gt_landmarks_weight = np.zeros((0, ), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
        assert len(gt_bboxes) == len(gt_landmarks)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            # landmark, 7 points, [x1,y1,x2,y2,x3,y3,x4,y4, lm1, lm2]
            landmarks=gt_landmarks,
            landmarks_weight=gt_landmarks_weight,  # 0/1
        )
        return ann



def build(image_set):
    cfg = mmcv.Config.fromfile('configs/datasets/lm_wider_face.py')
    from mmdet.datasets import build_dataset
    if image_set == 'val':
        dataset = build_dataset(cfg.data.test)
    elif image_set == 'train':
        dataset = build_dataset(cfg.data.train)
    return dataset