# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import numpy as np

from dataset.JointsDataset import JointsDataset
import dataset.panoDataset as pano

logger = logging.getLogger(__name__)


class PANO(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, state, transform=None):
        super().__init__(cfg, root, image_set, state, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        teeth_num = {"18": 1, "17": 2, "16": 3, "15": 4, "14": 5, "13": 6, "12": 7, "11": 8, "21": 9, "22": 10,
                     "23": 11, "24": 12, "25": 13, "26": 14, "27": 15, "28": 16, "48": 17, "47": 18, "46": 19,
                     "45": 20, "44": 21, "43": 22, "42": 23, "41": 24, "31": 25, "32": 26, "33": 27, "34": 28,
                     "35": 29, "36": 30, "37": 31, "38": 32}

        # deal with class names
        cats = ['teeth']
        self.classes = cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, [1]))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes
            ]
        )

        print('==> initializing pano {} data.'.format(state))
        self.annot_path = os.path.join('C:\\Users\CGIP\Desktop\github\CenterNet\data\pano', state)
        self.pano = pano.PANODataset(self.annot_path)

        # load image file names
        self.image_set_index = self.pano.getImgIds()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 32

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        data: [ {"num":1, "bbox":[(0,0),(0,0),(0,0),(0,0)], "center":(0,0)} , {} , ... ]
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        img, objs = self.pano.loadData(index)

        width = np.size(img, 1)
        height = np.size(img, 0)

        # sanitize bboxes
        clean_bbox = [0, 0, width, height]

        rec = []
        joints_3d = np.zeros((self.num_joints, 2), dtype=np.float)
        joints_3d_vis = np.zeros((self.num_joints, 2), dtype=np.float)
        visible = np.zeros((self.num_joints, 1), dtype=np.float)

        for ipt in range(self.num_joints):
            joints_3d[ipt, 0] = objs[ipt+1]["center"][0]
            joints_3d[ipt, 1] = objs[ipt+1]["center"][1]
            visible[ipt] = objs[ipt+1]['visible']

            joints_3d_vis[ipt, 0] = visible[ipt]
            joints_3d_vis[ipt, 1] = visible[ipt]

        center, scale = self._box2cs(clean_bbox[:4])
        rec.append({
            'image': img,
            'center': center,
            'scale': scale,
            'joints_3d': joints_3d,
            'joints_3d_vis': joints_3d_vis,
            'filename': index,
            'imgnum': 0,
            'visible': visible
        })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale