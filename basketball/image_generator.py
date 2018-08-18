# generate image from map and camera pose

import numpy as np
import cv2 as cv

class ImageGenerator:
    def __init__(self):
        s = 0.0254  #inch to meter
        # map border is 70 pixel
        # 1 pixel is 1 inch (1/12 foot) in map
        m1 = np.matrix([[1, 0, -70],
                       [0, 1, -70],
                       [0, 0, 1]])

        # flip Y direction
        m2 = np.matrix([[1, 0, 0],
                       [0, -1, 600],
                       [0, 0, 1]])

        # inch to meter
        m3 = np.matrix([[s, 0, 0],
                       [0, s, 0],
                       [0, 0, 1]])
        self.map_to_world = m3*m2*m1

    def _camera_to_homography(self, camera):
        """
        :param camera: 1 * 9 camera parameter
        :return: 3x3 matrix homography
        """
        u, v, f = camera[0], camera[1], camera[2]
        K = np.matrix([[f, 0, u], [0, f, v], [0, 0, 1]])
        rod = camera[3:6]   # rodurigues
        R = np.zeros((3, 3))
        cv.Rodrigues(rod, R)
        C = camera[6:9]
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
    # input: map an camera
    # ouput: a synthesized image
    generator = ImageGenerator()
    camera = np.array([640, 360, 2314.4, 1.7441, -0.3134, 0.2688, 13.0099, -14.8109, 6.1790])
    court_map = cv.imread('basketball_map.png')
    cv.imshow('map', court_map)
    image = generator.generate_image(camera, court_map)
    cv.imshow('warped map', image)
    cv.waitKey(0)



