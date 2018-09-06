"""generate synthesized image from basketball map and camera pose"""

import numpy as np
import cv2 as cv
import scipy.io as sio
from math import *
import scipy.signal as sig


class ImageGenerator:
    def __init__(self):
        s = 0.0254  """ inch to meter
        map border is 70 pixel
        1 pixel is 1 inch (1/12 foot) in map"""
        m1 = np.matrix([[1, 0, -70],
                        [0, 1, -70],
                        [0, 0, 1]])

        """flip Y direction"""
        m2 = np.matrix([[1, 0, 0],
                        [0, -1, 600],
                        [0, 0, 1]])

        """inch to meter"""
        m3 = np.matrix([[s, 0, 0],
                        [0, s, 0],
                        [0, 0, 1]])
        self.map_to_world = m3 * m2 * m1

    def _camera_to_homography(self, camera):
        """
        :param camera: 1 * 9 camera parameter
        :return: 3x3 matrix homography
        """
        u, v, f = camera[0], camera[1], camera[2]
        K = np.matrix([[f, 0, u], [0, f, v], [0, 0, 1]])

        pan, tilt = camera[3], camera[4]
        pan = radians(pan)
        tilt = radians(tilt)

        base = np.array([1.5804, -0.1186, 0.1249])
        base_rotation = np.zeros([3, 3])
        cv.Rodrigues(base, base_rotation)

        R = np.dot(np.array([[1, 0, 0],
                             [0, cos(tilt), sin(tilt)],
                             [0, -sin(tilt), cos(tilt)]]),
                   np.array([[cos(pan), 0, -sin(pan)],
                             [0, 1, 0],
                             [sin(pan), 0, cos(pan)]]))
        R = np.dot(R, base_rotation)

        # rod = camera[3:6]   # rodurigues
        # R = np.zeros((3, 3))
        # cv.Rodrigues(rod, R)
        C = camera[5:8]
        C = C.reshape((3, 1))
        t = np.matmul(-R, C)
        P = K.dot(np.hstack((R, t)))
        H = P[:, [0, 1, 3]]
        return H

    def generate_image(self, camera, map):
        """
        :param camera: 1*9, px, py, f, rotate, camera center
        :param map: an RGB image
        :return:    project RGB image
        """

        H_map_to_world = self.map_to_world
        H_world_to_image = generator._camera_to_homography(camera)
        H = H_world_to_image * H_map_to_world
        image = cv.warpPerspective(map, H, (1280, 720))
        return image


if __name__ == "__main__":

    """input: map an camera
    ouput: a synthesized image"""
    generator = ImageGenerator()

    seq = sio.loadmat("./basketball/basketball/basketball_anno.mat")
    annotation = seq["annotation"]

    court_map = cv.imread('./basketball/basketball_map.png')

    pan_arr = np.ndarray([annotation.size])
    tilt_arr = np.ndarray([annotation.size])
    f_arr = np.ndarray([annotation.size])
    for i in range(annotation.size):
        pan_arr[i], tilt_arr[i], f_arr[i] = annotation[0][i]['ptz'].squeeze()

    """smooth pan, tilt and f"""
    pan_arr = sig.savgol_filter(pan_arr, 181, 1)
    tilt_arr = sig.savgol_filter(tilt_arr, 181, 1)
    f_arr = sig.savgol_filter(f_arr, 181, 1)

    for i in range(annotation.size):
        pan, tilt, _ = annotation[0][i]['ptz'].squeeze()
        camera = np.array([640, 360, f_arr[i], pan_arr[i], tilt_arr[i], 13.0099, -14.8109, 6.1790])
        image = generator.generate_image(camera, court_map)

        image_name = "./basketball/basketball/synthesize_images/" + str(i) + ".jpg"
        cv.imwrite(image_name, image)
       
