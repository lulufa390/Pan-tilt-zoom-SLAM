import numpy as np
import cv2 as cv
import scipy.io as sio


class SequenceManager:
    def __init__(self, annotation_path, image_path, bounding_box_path=None):
        self.height = 720
        self.width = 1280

        """annotation data"""
        seq = sio.loadmat(annotation_path)
        self.annotation = seq["annotation"]
        self.anno_size = self.annotation.size
        meta = seq['meta']
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = meta[0][0]["cc"][0]

        self.image_path = image_path

        self.bounding_box = []
        if bounding_box_path:
            self.bounding_box = sio.loadmat(bounding_box_path)['bounding_box']

    def get_camera_center(self):
        return self.c

    def get_base_rotation(self):
        return self.base_rotation

    def get_basketball_image_gray(self, index):
        """
        :param index: image index for sequence
        :return: gray image
        """

        """for soccer!"""
        # img = cv.imread(self.image_path + "00000" + str(index + 515) + ".jpg")

        """for basketball!"""
        img = cv.imread(self.image_path + "/000" + str(index + 84000) + ".jpg")

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    def get_basketball_image(self, index):
        """
        :param index: image index for sequence
        :return: color image
        """

        """for soccer!"""
        # img = cv.imread(self.image_path + "00000" + str(index + 515) + ".jpg")

        """for basketball!"""
        img = cv.imread(self.image_path + "/000" + str(index + 84000) + ".jpg", 1)

        return img

    def get_bounding_box_mask(self, index):
        """
        function to get mask to remove features on players
        :param i:
        :return:
        """
        if len(self.bounding_box) > 0:
            tmp_mask = np.ones([self.height, self.width])
            for j in range(self.bounding_box[0][index].shape[0]):
                if self.bounding_box[0][index][j][4] > 0.6:
                    for x in range(int(self.bounding_box[0][index][j][0]),
                                   int(self.bounding_box[0][index][j][2])):
                        for y in range(int(self.bounding_box[0][index][j][1]),
                                       int(self.bounding_box[0][index][j][3])):
                            tmp_mask[y, x] = 0
            return tmp_mask

    def get_ptz(self, index):
        return self.annotation[0][index]['ptz'].squeeze()


if __name__ == '__main__':
    obj = SequenceManager("./basketball/basketball/basketball_anno.mat", "./basketball/basketball/images",
                          "./objects_basketball.mat")
    cv.imshow("test", obj.get_bounding_box_mask(100))
    cv.waitKey(0)
