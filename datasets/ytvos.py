#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import numpy as np
import random, mmcv, os
import os.path as osp
from pycocotools.ytvos import YTVOS
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose, to_tensor


@DATASETS.register_module
class VISDataset(CustomDataset):

	def __init__(self,
				 ann_file,
				 pipeline,
				 data_root=None,
				 img_prefix='',
				 seg_prefix=None,
				 proposal_file=None,
				 test_mode=False,
				 num_samples=None,
				 ignore_crowd=True,
				 ref_range=-1,
				 min_box_size=5,
				 aug_ref_bbox_param=None,
				 **kargs):

		self.ann_file = ann_file
		self.data_root = data_root
		self.img_prefix = img_prefix
		self.seg_prefix = seg_prefix
		self.proposal_file = proposal_file
		self.test_mode = test_mode
		self.ignore_crowd = ignore_crowd
		self.ref_range = ref_range
		self.min_box_size = min_box_size
		self.aug_ref_bbox_param = aug_ref_bbox_param

		# join paths if data_root is specified
		if self.data_root is not None:
			if not osp.isabs(self.ann_file):
				self.ann_file = osp.join(self.data_root, self.ann_file)
			if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
				self.img_prefix = osp.join(self.data_root, self.img_prefix)
			if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
				self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
			if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
				self.proposal_file = osp.join(self.data_root, self.proposal_file)

		# load annotations
		timer = mmcv.Timer()
		self.vid_infos = self.load_annotations(self.ann_file)
		img_ids = []
		for idx, vid_info in enumerate(self.vid_infos):
			for frame_id in range(len(vid_info['filenames'])):
				img_ids.append((idx, frame_id))
		self.img_ids = img_ids
		print("Loaded annotation times:", timer.since_start())

		# Limit the number of samples
		if num_samples is not None:
			self.img_ids = self.img_ids[:num_samples]
			print("Number of samples: ", num_samples)

		# filter images with no annotation during training
		if not test_mode:
			valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
				if len(self.get_ann_info(v, f)['bboxes'])]
			print("Number of valid_inds:", len(valid_inds))
			self.img_ids = [self.img_ids[i] for i in valid_inds]

		# set group flag for the sampler
		if not self.test_mode:
			self._set_group_flag()

		# processing pipeline
		self.pipeline = Compose(pipeline)

		# construct img_infos
		self.img_infos = []
		for (vid, frame_id) in self.img_ids:
			vid_info = self.vid_infos[vid]
			height = vid_info.get('height', None)
			width = vid_info.get('width', None)
			timestamp_micros = vid_info['timestamp_micros'][frame_id]
			filename = osp.join(self.img_prefix, vid_info['filenames'][frame_id])
			self.img_infos.append(dict(
				filename=filename,
				height=height, width=width,
				timestamp_micros=timestamp_micros))

	def __len__(self):
		return len(self.img_ids)

	def __getitem__(self, idx):
		if self.test_mode:
			return self.prepare_test_img(self.img_ids[idx])
		while True:
			data = self.prepare_train_img(self.img_ids[idx])
			if data is None:
				idx = self._rand_another(idx)
				continue
			return data

	def _set_group_flag(self):
		self.flag = np.ones(len(self), dtype='uint8')

	def load_annotations(self, ann_file):
		self.ytvos = YTVOS(ann_file)
		self.cat_ids = self.ytvos.getCatIds()
		self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}
		self.vid_ids = self.ytvos.getVidIds()

		vid_infos = []
		for i in self.vid_ids:
			info = self.ytvos.loadVids([i])[0]
			info['filenames'] = info['file_names']
			vid_infos.append(info)

		return vid_infos

	def get_ann_info(self, idx, frame_id):
		vid_id = self.vid_infos[idx]['id']
		ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
		ann_info = self.ytvos.loadAnns(ann_ids)
		return self._parse_ann_info(ann_info, frame_id)

	def bbox_aug(self, bbox, img_size):
		assert self.aug_ref_bbox_param is not None
		center_off = self.aug_ref_bbox_param[0]
		size_perturb = self.aug_ref_bbox_param[1]
		n_bb = bbox.shape[0]
		# bbox center offset
		center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
		# bbox resize ratios
		resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
		# bbox: x1, y1, x2, y2
		centers = (bbox[:,:2]+ bbox[:,2:])/2.
		sizes = bbox[:,2:] - bbox[:,:2]
		new_centers = centers + center_offs * sizes
		new_sizes = sizes * resize_ratios
		new_x1y1 = new_centers - new_sizes/2.
		new_x2y2 = new_centers + new_sizes/2.
		c_min = [0,0]
		c_max = [img_size[1], img_size[0]]
		new_x1y1 = np.clip(new_x1y1, c_min, c_max)
		new_x2y2 = np.clip(new_x2y2, c_min, c_max)
		bbox = np.hstack((new_x1y1, new_x2y2)).astype('float32')
		return bbox

	def sample_ref(self, idx):
		"""sample another frame in the same sequence as reference
		"""
		vid, frame_id = idx
		vid_info = self.vid_infos[vid]

		sample_range = list(range(len(vid_info['filenames'])))
		if self.ref_range != -1:
			start_idx = max([frame_id-self.ref_range, 0])
			end_idx = min([frame_id+self.ref_range+1, len(sample_range)])
			sample_range = [sample_range[idx]
				for idx in range(start_idx, end_idx) if idx!=frame_id]

		valid_samples = [
			(vid, i) for i in sample_range
			if (i != frame_id) and ((vid, i) in self.img_ids)
		]
		if len(valid_samples) == 0:
			return vid, frame_id
		else:
			return random.choice(valid_samples)

	def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
		gt_bboxes = []
		gt_labels = []
		gt_ids = []
		gt_bboxes_ignore = []
		if with_mask:
			gt_masks = []

		for i, ann in enumerate(ann_info):
			bbox = ann['bboxes'][frame_id]
			if 'areas' in ann:
				area = ann['areas'][frame_id]
			else:
				area = 1
			if 'segmentations' in ann:
				segm = ann['segmentations'][frame_id]
			else:
				segm = None

			if bbox is None:
				continue
			x1, y1, w, h = bbox
			if area <= 0 or w < 1 or h < 1:
				continue
			if min([h, w]) < self.min_box_size:
				continue
			bbox = [x1, y1, x1 + w - 1, y1 + h - 1]

			if self.ignore_crowd and ann.get('iscrowd', False):
				gt_bboxes_ignore.append(bbox)
			else:
				gt_bboxes.append(bbox)
				gt_ids.append(ann['id'])
				gt_labels.append(self.cat2label[ann['category_id']])

			if with_mask:
				gt_masks.append(segm)

		if gt_bboxes:
			gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
			gt_labels = np.array(gt_labels, dtype=np.int64)
		else:
			gt_bboxes = np.zeros((0, 4), dtype=np.float32)
			gt_labels = np.array([], dtype=np.int64)

		if gt_bboxes_ignore:
			gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
		else:
			gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

		ann = dict(
			bboxes=gt_bboxes,
			labels=gt_labels,
			obj_ids=gt_ids,
			bboxes_ignore=gt_bboxes_ignore)

		if with_mask:
			ann['masks'] = gt_masks

		return ann

	def prepare_train_img(self, idx):
		# prepare a pair of image in a sequence
		vid, frame_id = idx
		vid_info = self.vid_infos[vid]
		_, ref_frame_id = self.sample_ref(idx)

		# get image info
		height = vid_info.get('height', None)
		width = vid_info.get('width', None)
		filename = osp.join(self.img_prefix, vid_info['filenames'][frame_id])
		ref_filename = osp.join(self.img_prefix, vid_info['filenames'][ref_frame_id])
		img_info = dict(filename=filename, height=height, width=width)
		ref_img_info = dict(filename=ref_filename, height=height, width=width)

		# get annotation info
		ann_info = self.get_ann_info(vid, frame_id)
		ref_ann_info = self.get_ann_info(vid, ref_frame_id)

		# load results
		results = dict(img_info=img_info, ann_info=ann_info)
		self.pre_pipeline(results)
		results = self.pipeline(results)
		if len(results['gt_bboxes'].data) == 0:
			return None

		# load ref_results
		flip = results['img_metas'].data['flip']
		ref_results = dict(img_info=ref_img_info, ann_info=ref_ann_info, flip=flip)
		self.pre_pipeline(ref_results)
		ref_results = self.pipeline(ref_results)
		if len(ref_results['gt_bboxes'].data) == 0:
			return None

		# ref bbox transforms to simulate the non-ideal condition
		ref_bboxes = ref_results['gt_bboxes']
		ref_labels = ref_results['gt_labels']
		if self.aug_ref_bbox_param is not None:
			ref_img_shape = ref_results['img_metas'].data['img_shape']
			ref_bboxes = self.bbox_aug(ref_bboxes.data.numpy(), ref_img_shape)
			ref_bboxes = DC(to_tensor(ref_bboxes))

		# compute matching of reference frame with current frame
		gt_ids = ann_info['obj_ids']
		ref_ids = ref_ann_info['obj_ids']
		gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]

		# add more info to img_metas
		results['img_metas'].data['is_first'] = (frame_id == 0)
		results['img_metas'].data['ref_filename'] = ref_filename
		results['ref_img_metas'] = ref_results['img_metas']

		# add more info to results
		results['ref_img'] = ref_results['img']
		results['gt_ref_bboxes'] = ref_bboxes
		results['gt_ref_labels'] = ref_labels
		results['gt_ref_weights'] = ref_results['gt_weights']

		results['gt_pids'] = DC(to_tensor(gt_pids))
		results['gt_ids'] = DC(gt_ids, stack=False, cpu_only=True)
		results['ref_ids'] = DC(ref_ids, stack=False, cpu_only=True)
		return results

	def prepare_test_img(self, idx):
		# get image info
		vid, frame_id = idx
		vid_info = self.vid_infos[vid]
		height = vid_info.get('height', None)
		width = vid_info.get('width', None)
		filename = osp.join(self.img_prefix, vid_info['filenames'][frame_id])
		img_info = dict(filename=filename, height=height, width=width)

		# load results
		results = dict(img_info=img_info)
		self.pre_pipeline(results)
		results = self.pipeline(results)
		results['img_metas'][0].data['is_first'] = (frame_id == 0)
		return results







# import os.path as osp
# import random

# import mmcv
# import numpy as np
# import torch
# from mmcv.parallel import DataContainer as DC
# from pycocotools.ytvos import YTVOS
# from util.box_ops import box_xyxy_to_cxcywh

# # from datasets.extra_aug import ExtraAugmentation
# from mmdet.datasets import DATASETS, build_dataloader, build_dataset
# from mmdet.datasets.custom import CustomDataset
# # from pyson.utils import memoize

# from .transforms import (BboxTransform, ImageTransform, MaskTransform,
#                          Numpy2Tensor)

# from .utils import random_scale, to_tensor


# @DATASETS.register_module()
# class YTVOSDataset(CustomDataset):
#     CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard',
#                'sedan', 'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit',
#                'duck', 'cat', 'cow', 'fish', 'train', 'horse', 'turtle',
#                'bear', 'motorbike', 'giraffe', 'leopard', 'fox', 'deer', 'owl',
#                'surfboard', 'airplane', 'truck', 'zebra', 'tiger', 'elephant',
#                'snowboard', 'boat', 'shark', 'mouse', 'frog', 'eagle',
#                'earless_seal', 'tennis_racket')

#     def __init__(self,
#                  ann_file,
#                  img_prefix,
#                  img_scale,
#                  img_norm_cfg,
#                  size_divisor=None,
#                  proposal_file=None,
#                  num_max_proposals=1000,
#                  flip_ratio=0,
#                  with_mask=True,
#                  with_crowd=True,
#                  with_label=True,
#                  with_track=False,
#                  extra_aug=None,
#                  aug_ref_bbox_param=None,
#                  resize_keep_ratio=True,
#                  test_mode=False,
#                  num_images=None):
#         # prefix of images path
#         self.img_prefix = img_prefix

#         # load annotations (and proposals)
#         self.vid_infos = self.load_annotations(ann_file)
#         img_ids = []
#         for idx, vid_info in enumerate(self.vid_infos):
#             for frame_id in range(len(vid_info['filenames'])):
#                 img_ids.append((idx, frame_id))
#         self.img_ids = img_ids
#         if proposal_file is not None:
#             self.proposals = self.load_proposals(proposal_file)
#         else:
#             self.proposals = None
#         # filter images with no annotation during training
#         if not test_mode:
#             valid_inds = [
#                 i for i, (v, f) in enumerate(self.img_ids)
#                 if len(self.get_ann_info(v, f)['bboxes'])
#             ]

#             self.img_ids = [self.img_ids[i] for i in valid_inds]


#         if num_images is not None:
#             self.img_ids = self.img_ids[:num_images]
#         # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
#         self.img_scales = img_scale if isinstance(img_scale,
#                                                   list) else [img_scale]
#         assert mmcv.is_list_of(self.img_scales, tuple)
#         # normalization configs
#         self.img_norm_cfg = img_norm_cfg

#         # max proposals per image
#         self.num_max_proposals = num_max_proposals
#         # flip ratio
#         self.flip_ratio = flip_ratio
#         assert flip_ratio >= 0 and flip_ratio <= 1
#         # padding border to ensure the image size can be divided by
#         # size_divisor (used for FPN)
#         self.size_divisor = size_divisor

#         # with mask or not (reserved field, takes no effect)
#         self.with_mask = with_mask
#         # some datasets provide bbox annotations as ignore/crowd/difficult,
#         # if `with_crowd` is True, then these info is returned.
#         self.with_crowd = with_crowd
#         # with label is False for RPN
#         self.with_label = with_label
#         self.with_track = with_track
#         # params for augmenting bbox in the reference frame
#         self.aug_ref_bbox_param = aug_ref_bbox_param
#         # in test mode or not
#         self.test_mode = test_mode

#         # set group flag for the sampler
#         if not self.test_mode:
#             self._set_group_flag()
#         # transforms
#         self.img_transform = ImageTransform(size_divisor=self.size_divisor,
#                                             **self.img_norm_cfg)
#         self.bbox_transform = BboxTransform()
#         self.mask_transform = MaskTransform()
#         self.numpy2tensor = Numpy2Tensor()

#         # if use extra augmentation
#         # if extra_aug is not None:
#         #     self.extra_aug = ExtraAugmentation(**extra_aug)
#         # else:
#         #     self.extra_aug = None

#         # image rescale if keep ratio
#         self.resize_keep_ratio = resize_keep_ratio

#     def __len__(self):
#         return len(self.img_ids)

#     def __getitem__(self, idx):
#         if self.test_mode:
#             return self.prepare_test_img(self.img_ids[idx])
#         data = self.prepare_train_img(self.img_ids[idx])
#         return data

#     def load_annotations(self, ann_file):
#         def f():
#             ytvos = YTVOS(ann_file)
#             cat_ids = ytvos.getCatIds()
#             cat2label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
#             vid_ids = ytvos.getVidIds()
#             vid_infos = []
#             for i in vid_ids:
#                 info = ytvos.loadVids([i])[0]
#                 info['filenames'] = info['file_names']
#                 vid_infos.append(info)

#             return vid_infos, ytvos, cat_ids, cat2label, vid_ids

#         vid_infos, self.ytvos, self.cat_ids, self.cat2label, self.vid_ids = f()

#         return vid_infos

#     def get_ann_info(self, idx, frame_id):
#         vid_id = self.vid_infos[idx]['id']
#         ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
#         ann_info = self.ytvos.loadAnns(ann_ids)
#         return self._parse_ann_info(ann_info, frame_id)

#     def _set_group_flag(self):
#         """Set flag according to image aspect ratio.

#         Images with aspect ratio greater than 1 will be set as group 1,
#         otherwise group 0.
#         """
#         self.flag = np.zeros(len(self), dtype=np.uint8)
#         for i in range(len(self)):
#             vid_id, _ = self.img_ids[i]
#             vid_info = self.vid_infos[vid_id]
#             if vid_info['width'] / vid_info['height'] > 1:
#                 self.flag[i] = 1

#     def bbox_aug(self, bbox, img_size):
#         assert self.aug_ref_bbox_param is not None
#         center_off = self.aug_ref_bbox_param[0]
#         size_perturb = self.aug_ref_bbox_param[1]

#         n_bb = bbox.shape[0]
#         # bbox center offset
#         center_offs = (2 * np.random.rand(n_bb, 2) - 1) * center_off
#         # bbox resize ratios
#         resize_ratios = (2 * np.random.rand(n_bb, 2) - 1) * size_perturb + 1
#         # bbox: x1, y1, x2, y2
#         centers = (bbox[:, :2] + bbox[:, 2:]) / 2.
#         sizes = bbox[:, 2:] - bbox[:, :2]
#         new_centers = centers + center_offs * sizes
#         new_sizes = sizes * resize_ratios
#         new_x1y1 = new_centers - new_sizes / 2.
#         new_x2y2 = new_centers + new_sizes / 2.
#         c_min = [0, 0]
#         c_max = [img_size[1], img_size[0]]
#         new_x1y1 = np.clip(new_x1y1, c_min, c_max)
#         new_x2y2 = np.clip(new_x2y2, c_min, c_max)
#         bbox = np.hstack((new_x1y1, new_x2y2)).astype(np.float32)
#         return bbox

#     def sample_ref(self, idx):
#         # sample another frame in the same sequence as reference
#         vid, frame_id = idx
#         vid_info = self.vid_infos[vid]
#         sample_range = range(len(vid_info['filenames']))
#         valid_samples = []
#         for i in sample_range:
#             # check if the frame id is valid
#             ref_idx = (vid, i)
#             if i != frame_id and ref_idx in self.img_ids:
#                 valid_samples.append(ref_idx)
#         assert len(valid_samples) > 0
#         return random.choice(valid_samples)

#     def prepare_train_img(self, idx):
#         # prepare a pair of image in a sequence
#         vid, frame_id = idx
#         vid_info = self.vid_infos[vid]
#         # load image
#         img = mmcv.imread(
#             osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
#         basename = osp.basename(vid_info['filenames'][frame_id])
#         _, ref_frame_id = self.sample_ref(idx)
#         ref_img = mmcv.imread(
#             osp.join(self.img_prefix, vid_info['filenames'][ref_frame_id]))
#         # load proposals if necessary
#         if self.proposals is not None:  # default is None
#             proposals = self.proposals[idx][:self.num_max_proposals]
#             # TODO: Handle empty proposals properly. Currently images with
#             # no proposals are just ignored, but they can be used for
#             # training in concept.
#             if len(proposals) == 0:
#                 return None
#             if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
#                 raise AssertionError(
#                     'proposals should have shapes (n, 4) or (n, 5), '
#                     'but found {}'.format(proposals.shape))
#             if proposals.shape[1] == 5:
#                 scores = proposals[:, 4, None]
#                 proposals = proposals[:, :4]
#             else:
#                 scores = None

#         ann = self.get_ann_info(vid, frame_id)
#         ref_ann = self.get_ann_info(vid, ref_frame_id)
#         gt_bboxes = ann['bboxes']
#         gt_labels = ann['labels']
#         ref_labels = ref_ann['labels']
#         ref_bboxes = ref_ann['bboxes']
#         # obj ids attribute does not exist in current annotation
#         # need to add it
#         ref_ids = ref_ann['obj_ids']
#         gt_ids = ann['obj_ids']
#         # compute matching of reference frame with current frame
#         # 0 denote there is no matching
#         gt_pids = [ref_ids.index(i) + 1 if i in ref_ids else 0 for i in gt_ids]
#         # gt_pids = dict(
#         #     gt_ids=gt_ids,
#         #     ref_ids=ref_ids,
#         # )
#         if self.with_crowd:
#             gt_bboxes_ignore = ann['bboxes_ignore']

#         # skip the image if there is no valid gt bbox
#         if len(gt_bboxes) == 0:
#             return None

#         # extra augmentation
#         if self.extra_aug is not None:
#             img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
#                                                        gt_labels)

#         # apply transforms
#         flip = True if np.random.rand() < self.flip_ratio else False
#         img_scale = random_scale(self.img_scales)  # sample a scale
#         img, img_shape, pad_shape, scale_factor = self.img_transform(
#             img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
#         img = img.copy()
#         ref_img, ref_img_shape, _, ref_scale_factor = self.img_transform(
#             ref_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
#         ref_img = ref_img.copy()
#         if self.proposals is not None:
#             proposals = self.bbox_transform(proposals, img_shape, scale_factor,
#                                             flip)
#             proposals = np.hstack([proposals, scores
#                                    ]) if scores is not None else proposals
#         gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
#                                         flip)
#         ref_bboxes = self.bbox_transform(ref_bboxes, ref_img_shape,
#                                          ref_scale_factor, flip)
#         if self.aug_ref_bbox_param is not None:
#             ref_bboxes = self.bbox_aug(ref_bboxes, ref_img_shape)
#         if self.with_crowd:
#             gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
#                                                    scale_factor, flip)
#         if self.with_mask:
#             gt_masks = self.mask_transform(ann['masks'], pad_shape,
#                                            scale_factor, flip)

#         ori_shape = (vid_info['height'], vid_info['width'], 3)
#         img_meta = dict(ori_shape=ori_shape,
#                         img_shape=img_shape,
#                         pad_shape=pad_shape,
#                         scale_factor=scale_factor,
#                         flip=flip)

#         data = dict(img=DC(to_tensor(img), stack=True),
#                     ref_img=DC(to_tensor(ref_img), stack=True),
#                     img_meta=DC(img_meta, cpu_only=True),
#                     gt_bboxes=DC(to_tensor(gt_bboxes)),
#                     ref_bboxes=DC(to_tensor(ref_bboxes)),
#                     ref_labels=DC(to_tensor(ref_labels)))
#         if self.proposals is not None:
#             data['proposals'] = DC(to_tensor(proposals))
#         if self.with_label:
#             data['gt_labels'] = DC(to_tensor(gt_labels))
#         if self.with_track:
#             data['gt_pids'] = DC(gt_pids, cpu_only=True)
#         if self.with_crowd:
#             data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
#         if self.with_mask:
#             data['gt_masks'] = DC(gt_masks, cpu_only=True)
#         return data

#     def prepare_test_img(self, idx):
#         """Prepare an image for testing (multi-scale and flipping)"""
#         vid, frame_id = idx
#         vid_info = self.vid_infos[vid]
#         img = mmcv.imread(
#             osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
#         proposal = None

#         def prepare_single(img, frame_id, scale, flip, proposal=None):
#             _img, img_shape, pad_shape, scale_factor = self.img_transform(
#                 img, scale, flip, keep_ratio=self.resize_keep_ratio)
#             _img = to_tensor(_img)
#             _img_meta = dict(ori_shape=(vid_info['height'], vid_info['width'],
#                                         3),
#                              img_shape=img_shape,
#                              pad_shape=pad_shape,
#                              is_first=(frame_id == 0),
#                              video_id=vid,
#                              frame_id=frame_id,
#                              scale_factor=scale_factor,
#                              flip=flip)
#             if proposal is not None:
#                 if proposal.shape[1] == 5:
#                     score = proposal[:, 4, None]
#                     proposal = proposal[:, :4]
#                 else:
#                     score = None
#                 _proposal = self.bbox_transform(proposal, img_shape,
#                                                 scale_factor, flip)
#                 _proposal = np.hstack([_proposal, score
#                                        ]) if score is not None else _proposal
#                 _proposal = to_tensor(_proposal)
#             else:
#                 _proposal = None
#             return _img, _img_meta, _proposal

#         imgs = []
#         img_metas = []
#         proposals = []
#         for scale in self.img_scales:
#             _img, _img_meta, _proposal = prepare_single(
#                 img, frame_id, scale, False, proposal)
#             imgs.append(_img)
#             img_metas.append(DC(_img_meta, cpu_only=True))
#             proposals.append(_proposal)
#             if self.flip_ratio > 0:
#                 _img, _img_meta, _proposal = prepare_single(
#                     img, scale, True, proposal)
#                 imgs.append(_img)
#                 img_metas.append(DC(_img_meta, cpu_only=True))
#                 proposals.append(_proposal)
#         data = dict(img=imgs, img_meta=img_metas)
#         return data

#     def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
#         """Parse bbox and mask annotation.

#         Args:
#             ann_info (list[dict]): Annotation info of an image.
#             with_mask (bool): Whether to parse mask annotations.

#         Returns:
#             dict: A dict containing the following keys: bboxes, bboxes_ignore,
#                 labels, masks, mask_polys, poly_lens.
#         """
#         gt_bboxes = []
#         gt_labels = []
#         gt_ids = []
#         gt_bboxes_ignore = []
#         # Two formats are provided.
#         # 1. mask: a binary map of the same size of the image.
#         # 2. polys: each mask consists of one or several polys, each poly is a
#         # list of float.
#         if with_mask:
#             gt_masks = []
#             gt_mask_polys = []
#             gt_poly_lens = []
#         for i, ann in enumerate(ann_info):
#             # each ann is a list of masks
#             # ann:
#             # bbox: list of bboxes
#             # segmentation: list of segmentation
#             # category_id
#             # area: list of area
#             bbox = ann['bboxes'][frame_id]
#             area = ann['areas'][frame_id]
#             segm = ann['segmentations'][frame_id]
#             if bbox is None: continue
#             x1, y1, w, h = bbox
#             if area <= 0 or w < 1 or h < 1:
#                 continue
#             bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
#             if ann['iscrowd']:
#                 gt_bboxes_ignore.append(bbox)
#             else:
#                 gt_bboxes.append(bbox)
#                 gt_ids.append(ann['id'])
#                 gt_labels.append(self.cat2label[ann['category_id']])
#             if with_mask:
#                 gt_masks.append(self.ytvos.annToMask(ann, frame_id))
#                 mask_polys = [
#                     p for p in segm if len(p) >= 6
#                 ]  # valid polygons have >= 3 points (6 coordinates)
#                 poly_lens = [len(p) for p in mask_polys]
#                 gt_mask_polys.append(mask_polys)
#                 gt_poly_lens.extend(poly_lens)
#         if gt_bboxes:
#             gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
#             gt_labels = np.array(gt_labels, dtype=np.int64)
#         else:
#             gt_bboxes = np.zeros((0, 4), dtype=np.float32)
#             gt_labels = np.array([], dtype=np.int64)

#         if gt_bboxes_ignore:
#             gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
#         else:
#             gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

#         ann = dict(bboxes=gt_bboxes,
#                    labels=gt_labels,
#                    obj_ids=gt_ids,
#                    bboxes_ignore=gt_bboxes_ignore)

#         if with_mask:
#             ann['masks'] = gt_masks
#             # poly format is not used in the current implementation
#             ann['mask_polys'] = gt_mask_polys
#             ann['poly_lens'] = gt_poly_lens
#         return ann


# # class DETRWraper_YTVOSDataset(torch.utils.data.Dataset):
# #     def __init__(self, dataset):
# #         if isinstance(dataset, dict):
# #             self.dataset = build_dataset(dataset)
# #         else:
# #             self.dataset = dataset
# #         if hasattr(self.dataset, 'flag'):
# #             self.flag = self.dataset.flag
            

# #     def __len__(self):
# #         return len(self.dataset)

# #     def __getitem__(self, idx):
# #         item = self.dataset.__getitem__(idx)
# #         meta = 'img_meta'
# #         h, w, _ = item[meta].data['pad_shape']
# #         scale = torch.Tensor([w, h, w, h])

# #         target = dict(
# #             boxes=box_xyxy_to_cxcywh(item['gt_bboxes'].data)/ scale,
# #             labels=item['gt_labels'].data,
# #             orig_size=item[meta].data['ori_shape'],
# #             size=(h, w),
# #             iscrowd=False,
# #         )

# #         ref_target = dict(
# #             boxes=box_xyxy_to_cxcywh(item['ref_bboxes'].data)/ scale,
# #             labels=item['ref_labels'].data,
# #             orig_size=item[meta].data['ori_shape'],
# #             size=(h, w),
# #             iscrowd=False,
# #         )

# #         target[meta] = item[meta]
# #         item['target'] = DC(target)
# #         item['ref_target'] = DC(ref_target)
# #         return item

# # @memoize
# # def build(image_set, cfg_data):
# #     dataset_train = build_dataset(cfg_data[image_set])
# #     if image_set == 'train':
# #         dataset_train = DETRWraper_YTVOSDataset(dataset_train)
# #     print('Finish build dataloader YTVOS', dataset_train)
# #     return dataset_train
