"""
Functions to deal with a video sequence

Created by Luke, 2018.9

"""

import numpy as np
import cv2 as cv
import scipy.io as sio
from util import *
from image_process import *


class SequenceManager:
    def __init__(self, annotation_path, image_path, ground_truth_path, bounding_box_path=None):
        self.height = 720
        self.width = 1280

        # annotation data
        seq = sio.loadmat(annotation_path)
        annotation = seq["annotation"]
        meta = seq['meta']
        self.u, self.v = annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = meta[0][0]["cc"][0]

        self.image_path = image_path

        self.ground_truth_pan, self.ground_truth_tilt, self.ground_truth_f = load_camera_pose(ground_truth_path)

        self.anno_size = len(self.ground_truth_pan)

        self.bounding_box = []
        if bounding_box_path:
            self.bounding_box = sio.loadmat(bounding_box_path)['bounding_box']

    def get_camera_center(self):
        # interface for camera center
        return self.c

    def get_base_rotation(self):
        # interface for camera base rotation
        return self.base_rotation

    def get_image_gray(self, index, dataset_type=0):
        """
        :param index: image index for sequence
        :param dataset_type: 0 for basketball dataset, 1 for soccer dataset, 2 for synthesized court sequence
        :return: gray image
        """

        if dataset_type == 0:
            img = cv.imread(self.image_path + "/000" + str(index + 84000) + ".jpg", 1)
        elif dataset_type == 1:
            img = cv.imread(self.image_path + "/00000" + str(index + 515) + ".jpg")
        elif dataset_type == 2:
            img = cv.imread(self.image_path + "/" + str(index) + ".jpg")
        else:
            print("Unknown dataset!")
            return None

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        assert isinstance(img_gray, np.ndarray)

        return img_gray

    def get_image(self, index, dataset_type=0):
        """
        :param index: image index for sequence
        :param dataset_type: 0 for basketball dataset, 1 for soccer dataset, 2 for synthesized court sequence
        :return: color image
        """

        if dataset_type == 0:
            img = cv.imread(self.image_path + "/000" + str(index + 84000) + ".jpg", 1)
        elif dataset_type == 1:
            img = cv.imread(self.image_path + "/00000" + str(index + 515) + ".jpg")
        elif dataset_type == 2:
            img = cv.imread(self.image_path + "/" + str(index) + ".jpg")
        else:
            print("Unknown dataset!")
            return None

        assert isinstance(img, np.ndarray)

        return img

    def get_bounding_box_mask(self, index):
        """
        function to get mask to remove features on players
        :param index: image index for sequence
        :return: a player mask for that frame (1 for no player, 0 for player)
        """
        if len(self.bounding_box) > 0:
            tmp_mask = np.ones([self.height, self.width])

            # this only for soccer
            # tmp_mask[60:100, 60:490] = 0

            for j in range(self.bounding_box[0][index].shape[0]):
                if self.bounding_box[0][index][j][4] > 0.6:
                    for x in range(int(self.bounding_box[0][index][j][0]),
                                   int(self.bounding_box[0][index][j][2])):
                        for y in range(int(self.bounding_box[0][index][j][1]),
                                       int(self.bounding_box[0][index][j][3])):
                            tmp_mask[y, x] = 0
            return tmp_mask

    def get_ptz(self, index):
        return self.ground_truth_pan[index], self.ground_truth_tilt[index], self.ground_truth_f[index]


def ut_camera_center_and_base_rotation():
    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    print(input.get_camera_center())
    print(input.get_base_rotation())


def vis_keypoints():
    """
    This function is used to generate player detection result figure in paper.
    """

    # obj = SequenceManager("./basketball/basketball/basketball_anno.mat", "./basketball/basketball/images",
    #                       "./objects_basketball.mat")

    obj = SequenceManager("./basketball/basketball/basketball_anno.mat", "./seq3_blur/",
                          "./objects_soccer.mat")

    img = obj.get_image(192, 1)
    img2 = np.copy(img)

    mask = obj.get_bounding_box_mask(192)

    kp = detect_sift(obj.get_image_gray(192, 1), 400)
    # kp = detect_harris_corner_grid(obj.get_image_gray(0, 1), 4, 4)
    remove_index = remove_player_feature(kp, mask)
    after_kp = kp[remove_index]

    chacha_length = 7

    for i in range(len(kp)):
        x = int(kp[i][0])
        y = int(kp[i][1])
        cv.line(img, (x - chacha_length, y - chacha_length), (x + chacha_length, y + chacha_length),
                color=(0, 255, 255),
                thickness=2)
        cv.line(img, (x - chacha_length, y + chacha_length), (x + chacha_length, y - chacha_length),
                color=(0, 255, 255),
                thickness=2)

    for i in range(len(after_kp)):
        x = int(after_kp[i][0])
        y = int(after_kp[i][1])
        cv.line(img2, (x - chacha_length, y - chacha_length), (x + chacha_length, y + chacha_length),
                color=(0, 255, 255),
                thickness=2)
        cv.line(img2, (x - chacha_length, y + chacha_length), (x + chacha_length, y - chacha_length),
                color=(0, 255, 255),
                thickness=2)

    boxes = obj.bounding_box[0][192]
    for j in range(boxes.shape[0]):
        if boxes[j][4] > 0.6:
            cv.line(img2, (boxes[j][0], boxes[j][1]),
                    (boxes[j][2], boxes[j][1]), color=(255, 255, 255), thickness=5)
            cv.line(img2, (boxes[j][0], boxes[j][1]),
                    (boxes[j][0], boxes[j][3]), color=(255, 255, 255), thickness=5)
            cv.line(img2, (boxes[j][0], boxes[j][3]),
                    (boxes[j][2], boxes[j][3]), color=(255, 255, 255), thickness=5)
            cv.line(img2, (boxes[j][2], boxes[j][1]),
                    (boxes[j][2], boxes[j][3]), color=(255, 255, 255), thickness=5)

    cv.imshow("test2", img2)
    cv.imshow("test", img)

    cv.imwrite("./img_keypoints.jpg", img)
    cv.imwrite("./img_keypoints_detection.jpg", img2)

    cv.waitKey(0)



if __name__ == '__main__':
    # ut_camera_center_and_base_rotation()

    vis_keypoints()

    # obj = SequenceManager("./basketball/basketball/basketball_anno.mat", "./basketball/basketball/images",
    #                       "./objects_basketball.mat")

    obj = SequenceManager("./two_point_calib_dataset/highlights/seq3_anno.mat","./seq3_blur",
                   "./objects_soccer.mat")


    pan = np.ndarray([1000])
    tilt = np.ndarray([1000])
    f = np.ndarray([1000])

    pre_pan, pre_tilt, pre_f = obj.get_ptz(0)
    for i in range(1, 73):
        now_pan, now_tilt, now_f = obj.get_ptz(i)

        delta_pan = (now_pan - pre_pan) / 6
        delta_tilt = (now_tilt - pre_tilt) / 6
        delta_f = (now_f - pre_f) / 6

        pan[(i-1)*6], tilt[(i-1)*6], f[(i-1)*6] = pre_pan, pre_tilt, pre_f
        pan[(i - 1) * 6 + 1], tilt[(i - 1) * 6 + 1], f[(i - 1) * 6 + 1] = pre_pan + delta_pan, pre_tilt + delta_tilt, pre_f + delta_f
        pan[(i - 1) * 6 + 2], tilt[(i - 1) * 6 + 2], f[
            (i - 1) * 6 + 2] = pre_pan + 2*delta_pan, pre_tilt + 2*delta_tilt, pre_f + 2*delta_f



        pan[(i - 1) * 6 + 3], tilt[(i - 1) * 6 + 3], f[
            (i - 1) * 6 + 3] = pre_pan + 3 * delta_pan, pre_tilt + 3 * delta_tilt, pre_f + 3 * delta_f
        pan[(i - 1) * 6 + 4], tilt[(i - 1) * 6 + 4], f[
            (i - 1) * 6 + 4] = pre_pan + 4 * delta_pan, pre_tilt + 4 * delta_tilt, pre_f + 4 * delta_f
        pan[(i - 1) * 6 + 5], tilt[(i - 1) * 6 + 5], f[
            (i - 1) * 6 + 5] = pre_pan + 5 * delta_pan, pre_tilt + 5 * delta_tilt, pre_f + 5 * delta_f


        pre_pan, pre_tilt, pre_f = now_pan, now_tilt, now_f
        print(i)

    save_camera_pose(pan[0:330], tilt[0:330], f[0:330], ".", "soccer3_ground_truth.mat")

    # cv.imshow("test", obj.get_bounding_box_mask(100))
    # cv.waitKey(0)
