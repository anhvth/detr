from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.registry import PIPELINES

# @PIPELINES.register_module
# class DefaultFormatBundleWithLandMark(object):
#     """Default formatting bundle.

# 	It simplifies the pipeline of formatting common fields, including "img",
# 	"proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
# 	These fields are formatted as follows.

# 	- img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
# 	- proposals: (1)to tensor, (2)to DataContainer
# 	- gt_bboxes: (1)to tensor, (2)to DataContainer
# 	- gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
# 	- gt_labels: (1)to tensor, (2)to DataContainer
# 	- gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
# 	- gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
# 					   (3)to DataContainer (stack=True)
# 	"""
#     def __call__(self, results):
#         if 'img' in results:
#             img = results['img']
#             if len(img.shape) < 3:
#                 img = np.expand_dims(img, -1)
#             img = np.ascontiguousarray(img.transpose(2, 0, 1))
#             results['img'] = DC(to_tensor(img), stack=True)
#         if 'prev_imgs' in results:
#             prev_imgs = results['prev_imgs']
#             for i, img in enumerate(prev_imgs):
#                 img = np.ascontiguousarray(img.transpose(2, 0, 1))
#                 prev_imgs[i] = img
#             prev_imgs = np.stack(prev_imgs)
#             results['prev_imgs'] = DC(to_tensor(prev_imgs), stack=True)

#         for key in [
#                 'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
#                 'gt_weights', 'gt_landmarks', 'gt_landmarks_weight',
#         ]:
#             if key not in results:
#                 continue
#             results[key] = DC(to_tensor(results[key]))
#         if 'gt_masks' in results:
#             results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
#         if 'gt_semantic_seg' in results:
#             results['gt_semantic_seg'] = DC(to_tensor(
#                 results['gt_semantic_seg'][None, ...]),
#                                             stack=True)
#         if 'gt_semsegs' in results:
#             gt_semsegs = np.ascontiguousarray(
#                 results['gt_semsegs'].astype('uint8'))
#             results['gt_semsegs'] = DC(to_tensor(gt_semsegs), stack=True)
#         return results

#     def __repr__(self):
#         return self.__class__.__name__
