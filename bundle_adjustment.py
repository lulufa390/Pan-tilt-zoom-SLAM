"""
Bundle Adjustment function
"""

import scipy.io as sio
import cv2 as cv
import numpy as np


class BundleAdjust:
    def __init__(self, annotation_path, image_path, bounding_box_path):
        self.width = 1280
        self.height = 720

        seq = sio.loadmat(annotation_path)
        self.annotation = seq["annotation"]
        self.meta = seq['meta']

        """base parameters"""
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        """image path"""
        self.image_path = image_path

        """bounding boxes"""
        bounding_box_data = sio.loadmat(bounding_box_path)
        self.bounding_box = bounding_box_data['bounding_box']

    def load_bounding_box_mask(self, i):
        """
        function to get mask to remove features on players
        :param i:
        :return:
        """
        tmp_mask = np.ones([self.height, self.width])
        for j in range(self.bounding_box[0][i].shape[0]):
            if self.bounding_box[0][i][j][4] > 0.6:
                for x in range(int(self.bounding_box[0][i][j][0]),
                               int(self.bounding_box[0][i][j][2])):
                    for y in range(int(self.bounding_box[0][i][j][1]),
                                   int(self.bounding_box[0][i][j][3])):
                        tmp_mask[y, x] = 0

        return tmp_mask

    def get_image_gray(self, index):
        """
        :param index: image index for video sequence
        :return: gray image
        """
        img = cv.imread(self.image_path + self.annotation[0][index]['image_name'][0])
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    def get_ptz(self, index):
        return self.annotation[0][index]['ptz'].squeeze()

    def bundle_adj(self, image_sequence):
        camera_pose = []
        images = []
        masks = []
        for i in image_sequence:
            camera_pose.append(self.get_ptz(i))
            images.append(self.get_image_gray(i))
            masks.append(self.load_bounding_box_mask(i))

        """using detect_compute_sift function to get SIFT features"""

        """
        Bundle Adjustment...
        """


if __name__ == '__main__':
    BundleAdjust_obj = BundleAdjust(...)

    img_seq = [0, 200, 550, 600, 650, 700]
    BundleAdjust_obj.bundle_adj(img_seq)