from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import glob

class PANODataset:
    def __init__(self, path):
        super(PANODataset, self).__init__()
        self.data_dir = path
        self.w_pad = 100
        self.h_pad = 0
        self.ratio = 2
        self.img_id =0
        self.num_joints = 32

    def getImgIds(self):
        file_list = glob.glob(self.data_dir + '/*.jpg')
        return file_list

    def _loadImg(self, file):
        return cv2.imread(file, cv2.IMREAD_COLOR)

    def loadData(self, file):
        original_img = self._loadImg(file)
        anns, mask = self._loadAnn(file)
        img, anns = self._cropImg(original_img, anns)
        img, anns = self._resize(img, anns)
        img = self._normalize(img)

        #mask = self._normalize(self._resize(self._cropImg(mask)))
        #cv2.imshow('result', mask)
        #cv2.waitKey(0)

        return img, anns

    def show_results(self, debugger, original_image, anns, save=True):
        if save==True:
            img_id = self.img_id
        else:
            img_id = 'ctdet'

        debugger.add_img(original_image, img_id=img_id)
        for ann in anns:
            bbox = ann["bbox"]
            center = ann["center"]
            bbox, center = self.getOriginalCoord(original_image, bbox, center)
            debugger.add_coco_bbox(bbox[:4], ann["category_id"] - 1, 1, img_id=img_id, teeth_num=ann["num"])
            debugger.add_center_point(center, img_id=img_id)

        if save==True:
            debugger.save_img(imgId=img_id, path='../data/pano/test_ann/')
            self.img_id += 1
        else:
            debugger.show_all_imgs(pause=True)

    def getOriginalCoord(self, original, center):
        width = np.size(original, 1)
        height = np.size(original, 0)

        original = original[self.h_pad:height - self.h_pad, self.w_pad:width - self.w_pad, :]

        width = np.size(original, 1)
        height = np.size(original, 0)

        if height * self.ratio < width:
            newH = height
            newW = height * self.ratio
        elif height * self.ratio > width:
            newH = int(width / self.ratio)
            newW = newH * self.ratio
        else:
            newH = height
            newW = width

        diffH = int((height - newH) / 2)
        diffW = int((width - newW) / 2)

        original = original[diffH:diffH + newH, diffW:diffW + newW, :]

        width = np.size(original, 1)
        height = np.size(original, 0)

        fx = width / 512
        fy = height / 256

        center = (center * np.array([fx, fy])) + np.array([diffW + self.w_pad, diffH + self.h_pad])

        return center

    def _normalize(self, img):
        inp = (img.astype(np.float32) / 255.)

        val = np.reshape(inp, (-1,3))
        mean = np.mean(val, axis=0).reshape(1, 1, 3)
        std = np.std(val, axis=0).reshape(1, 1, 3)

        inp = (inp - mean) / std
        return inp

    def _resize(self, img, anns=None):
        img_new = cv2.resize(img, (512, 256), interpolation=cv2.INTER_CUBIC)
        fx = np.size(img, 1) / 512
        fy = np.size(img, 0) / 256

        if anns==None:
            return img_new

        for ipt in range(1, self.num_joints+1):
            anns[ipt]["bbox"] = (anns[ipt]["bbox"]/np.array([fx, fy, fx, fy])).astype(int)
            anns[ipt]["center"] = (anns[ipt]["center"]/np.array([fx, fy])).astype(int)

        return img_new, anns

    def _cropImg(self, img, anns=None):
        width = np.size(img, 1)
        height = np.size(img, 0)

        img = img[self.h_pad:height-self.h_pad, self.w_pad:width-self.w_pad, :]

        width = np.size(img, 1)
        height = np.size(img, 0)

        if height*self.ratio<width:
            newH = height
            newW = height*self.ratio
        elif height*self.ratio>width:
            newH = int(width/self.ratio)
            newW = newH*self.ratio
        else:
            newH = height
            newW = width

        diffH = int((height-newH)/2)
        diffW = int((width-newW)/2)
        img = img[diffH:diffH+newH,diffW:diffW+newW,:]

        if anns==None:
            return img

        for ipt in range(1, self.num_joints+1):
            anns[ipt]["bbox"] = anns[ipt]["bbox"] - np.array([diffW + self.w_pad, diffH + self.h_pad, diffW + self.w_pad, diffH + self.h_pad])
            anns[ipt]["center"] = anns[ipt]["center"] - np.array([diffW + self.w_pad, diffH + self.h_pad])

        return img, anns

    def _loadAnn(self, file):
        '''
        :param file: pano jpg image path
        :return data: { 1 : {"bbox":[(0,0),(0,0),(0,0),(0,0)], "center":(0,0), "visible":1/0} , 2: {} , ... }
        '''
        data = {}

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        centerX = int(np.size(img, 1) / 2)
        centerY = int(np.size(img, 0) / 2)

        f = open(file.replace('jpg', 'txt'), 'r')
        txt = f.readlines()
        f.close()

        for idx, t in enumerate(txt):
            dict = {}
            t = t.replace('\n', '').split(', ')

            x_coord = []
            y_coord = []
            coords = np.zeros((8, 2), np.int32)

            for i in range(8):
                coords[i] = [centerX + int(t[2 * i + 3]), centerY + int(t[2 * i + 4])]
                x_coord.append(centerX + int(t[2 * i + 3]))
                y_coord.append(centerY + int(t[2 * i + 4]))

            bboxcoords = [min(x_coord), min(y_coord), max(x_coord), max(y_coord)]  # minX, minY, maxX, maxY

            dict["bbox"] = np.array([bboxcoords[0], bboxcoords[1], bboxcoords[2], bboxcoords[3]])  # from top-left corner, clockwise
            dict["center"] = np.array([centerX + int(t[27]), centerY + int(t[28])])

            if t[1]=='True':
                dict["visible"] = 1
            else:
                dict["visible"] = 0

            dict["mask"] = coords

            data[teeth_num[t[0]]] = dict

        mask = self.getSegMask(np.zeros_like(img), data)

        return data, mask

    def getSegMask(self, img, data):
        for i in range(1, 33):
            visible = data[i]["visible"]
            mask = data[i]["mask"]
            if visible==1:
                cv2.fillConvexPoly(img, mask, (255,0,0))

        return img

    def showImage(self, img, anns):
        for ann in anns:
            bbox = ann["bbox"]
            center = ann["center"]
            num = ann["num"]
            category_id = ann["category_id"]

            cv2.line(img, (bbox[0],bbox[1]), (bbox[2],bbox[1]), (0, 255, 0), 1)
            cv2.line(img, (bbox[2], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.line(img, (bbox[2], bbox[3]), (bbox[0], bbox[3]), (0, 255, 0), 1)
            cv2.line(img, (bbox[0], bbox[3]), (bbox[0], bbox[1]), (0, 255, 0), 1)

            cv2.putText(img, num, (center[0], center[1]), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
            if category_id == 1:
                cv2.circle(img, (center[0], center[1]), 3, (0, 0, 255), -1)
            else:
                cv2.circle(img, (center[0], center[1]), 3, (255, 0, 0), -1)

        cv2.imshow("result", img)
        cv2.waitKey(0)

teeth_num = {"18":1, "17":2, "16":3, "15":4, "14":5, "13":6, "12":7, "11":8, "21":9, "22":10, "23":11, "24":12, "25":13,
             "26":14, "27":15, "28":16, "48":17, "47":18, "46":19, "45":20, "44":21, "43":22, "42":23, "41":24,
             "31":25, "32":26, "33":27, "34":28, "35":29, "36":30, "37":31, "38":32}

if __name__ == "__main__":
    pano = PANODataset('E:\\dataset\\0210-anonymous\\train')
    file_lists = pano.getImgIds()
    for file in file_lists:
        pano.loadData(file)

