"""
A general pinhole camera model.

Created by Luke, 2019.1
"""

import cv2 as cv
import numpy as np
import glob
import scipy.io as sio


class PinholeCamera:
    def __init__(self, principal_point, focal_length, rotation, camera_center):
        self.principal_point = principal_point
        self.focal_length = focal_length
        self.camera_center = camera_center

        assert rotation.shape == (3, 3) or rotation.shape == (3,)
        if rotation.shape == (3, 3):
            self.rotation = rotation
        elif rotation.shape == (3,):
            self.rotation = np.zeros((3, 3))
            cv.Rodrigues(rotation, self.rotation)

    def set_principal_point(self, pp):
        assert type(pp) == np.ndarray and pp.shape == (2,)
        self.principal_point == pp

    def project_3d_point(self, p):
        K = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])

        point2d = np.dot(K, np.dot(self.rotation, p - self.camera_center))

        return point2d[0] / point2d[2], point2d[1] / point2d[2]


def ut_hockey_before_optimize_visualize():
    files = glob.glob("../../ice_hockey_1/olympic_2010_reference_frame/annotation/*.txt")
    N = len(files)

    annotation = sio.loadmat("../../ice_hockey_1/olympic_2010_reference_frame.mat")
    filename = annotation["images"]

    hockey_model = sio.loadmat("../../ice_hockey_1/ice_hockey_model.mat")
    points = hockey_model['points']
    line_index = hockey_model['line_segment_index']

    for i in range(N):
        file_name = files[i]

        data = np.loadtxt(file_name, delimiter='\t', skiprows=2)

        camera = PinholeCamera(data[0:2], data[2], data[3:6], data[6:9])

        img = cv.imread("../../ice_hockey_1/olympic_2010_reference_frame/image/" + filename[i])

        image_points = np.ndarray([len(points), 2])

        for j in range(len(points)):
            p = np.array([points[j][0], points[j][1], 0])

            image_points[j][0], image_points[j][1] = camera.project_3d_point(p)

        # draw lines
        for j in range(len(line_index)):
            begin = line_index[j][0]
            end = line_index[j][1]

            cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                    (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 2)

        cv.imshow("result", img)
        cv.waitKey(0)


if __name__ == "__main__":
    ut_hockey_before_optimize_visualize()
