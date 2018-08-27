"""
Bundle Adjustment tested on synthesized rays and basketball camera poses
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import scipy.io as sio
import cv2 as cv
from transformation import TransFunction
import scipy.signal as sig
import random
import matplotlib.pyplot as plt


class BundleAdjust:
    """Python class for Bundle Adjustment"""

    def __init__(self, model_path, annotation_path, data_path):
        """
        :param model_path: path for basketball court model
        :param annotation_path: path for camera sequence information
        :param data_path: path for synthesized points
        """

        random.seed(1)
        self.width = 1280
        self.height = 720

        """
        load the data of soccer field
        load annotation (ground truth)
        load synthesized features
        """
        basketball_model = sio.loadmat(model_path)
        self.line_index = basketball_model['line_segment_index']
        self.points = basketball_model['points']

        seq = sio.loadmat(annotation_path)
        self.annotation = seq["annotation"]
        self.meta = seq['meta']

        """ this is synthesized rays to generate 2d-point. Real data does not have this variable """
        data = sio.loadmat(data_path)
        self.ground_truth_ray = data["rays"]

        """
        initialize the fixed parameters of our algorithm
        u, v, base_rotation and c
        """
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        """select some frames from sequence"""
        self.key_frame = np.array([i for i in range(0, 3600, 100)])

        """ground_truth_x is a 1d array for ground truth camera pose and rays"""
        self.ground_truth_x = np.ndarray([3 * len(self.key_frame) + 2 * len(self.ground_truth_ray)])
        for i in range(len(self.key_frame)):
            self.ground_truth_x[3 * i], self.ground_truth_x[3 * i + 1], self.ground_truth_x[3 * i + 2] \
                = self.annotation[0][self.key_frame[i]]['ptz'].squeeze()

        for i in range(len(self.ground_truth_ray)):
            self.ground_truth_x[3 * len(self.key_frame) + 2 * i] = self.ground_truth_ray[i][0]
            self.ground_truth_x[3 * len(self.key_frame) + 2 * i + 1] = self.ground_truth_ray[i][1]

        """camera with gauss noise"""
        self.cameraArray = np.ndarray([3 * len(self.key_frame)])
        for i in range(len(self.key_frame)):
            self.cameraArray[3 * i] = self.ground_truth_x[3 * i] + random.gauss(0, 1)
            self.cameraArray[3 * i + 1] = self.ground_truth_x[3 * i + 1] + random.gauss(0, 1)
            self.cameraArray[3 * i + 2] = self.ground_truth_x[3 * i + 2] + random.gauss(0, 20)

        """rays with gauss noise"""
        self.ray3d = np.ndarray([2 * len(self.ground_truth_ray)])
        for i in range(len(self.ground_truth_ray)):
            self.ray3d[2 * i] = self.ground_truth_ray[i][0] + random.gauss(0, 1)
            self.ray3d[2 * i + 1] = self.ground_truth_ray[i][1] + random.gauss(0, 1)

        """generate image sequence"""
        self.image_list = []
        self.point_index_list = []
        for i in range(len(self.key_frame)):
            tmp_image, tmp_index = self.generate_image(i)
            self.image_list.append(tmp_image)
            self.point_index_list.append(tmp_index)

    def get_ground_truth_camera(self, index):
        """
        :param index: frame index
        :return: ground truth pan, tilt, f
        """
        return np.array([self.ground_truth_x[3 * index],
                         self.ground_truth_x[3 * index + 1], self.ground_truth_x[3 * index + 2]])

    def generate_image(self, index):
        """
        :param index: index for image
        :return: all points in image and their index in all rays
        """

        pan, tilt, f = self.get_ground_truth_camera(index)
        points = np.ndarray([0, self.ground_truth_ray.shape[1]])
        inner_index = np.ndarray([0])

        for i in range(len(self.ground_truth_ray)):
            x, y = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, self.ground_truth_ray[i][0],
                                                     self.ground_truth_ray[i][1])

            # x += random.gauss(0, 2)
            # y += random.gauss(0, 2)

            if 0 < x < self.width and 0 < y < self.height:
                points = np.row_stack([points, np.array([x, y])])
                inner_index = np.append(inner_index, i)

        return points, inner_index

    def get_image(self, i):
        """
        :param i: image index
        :return: get that image points and point index
        """
        return self.image_list[i], self.point_index_list[i]

    def get_observation_from_rays(self, pan, tilt, f, rays, ray_index):
        """
        :param pan:
        :param tilt:
        :param f:
        :param rays: all rays
        :param ray_index: inner index for all rays
        :return: 2d point for these rays
        """
        points = np.ndarray([0, 2])

        for j in range(len(ray_index)):
            theta = rays[int(ray_index[j])][0]
            phi = rays[int(ray_index[j])][1]
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, theta, phi)
            points = np.row_stack([points, np.asarray(tmp)])

        return points

    def fun(self, params, n_cameras, n_points):
        """
        :param params: contains camera parameters and rays
        :param n_cameras: number of camera poses
        :param n_points: number of rays
        :return: 1d residual
        """
        camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
        points_3d = params[n_cameras * 3:].reshape((n_points, 2))

        residual = np.ndarray([0])

        for i in range(n_cameras):
            point_2d, inner_index = self.get_image(i)
            if i == 0:
                ground_truth = self.get_ground_truth_camera(0)
                proj_point = self.get_observation_from_rays(ground_truth[0], ground_truth[1], ground_truth[2]
                                                            , points_3d, inner_index)
            else:
                proj_point = self.get_observation_from_rays(
                    camera_params[i, 0], camera_params[i, 1], camera_params[i, 2], points_3d, inner_index)

            residual = np.append(residual, proj_point.ravel() - point_2d.ravel())

        return residual

    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """

        x0 = np.hstack((self.cameraArray, self.ray3d))

        plt.figure(num="ground pan")
        plt.plot(self.ground_truth_x[:3 * len(self.key_frame)].reshape((-1, 3))[:, 0])

        plt.figure(num="ground tilt")
        plt.plot(self.ground_truth_x[:3 * len(self.key_frame)].reshape((-1, 3))[:, 1])

        plt.figure(num="ground f")
        plt.plot(self.ground_truth_x[:3 * len(self.key_frame)].reshape((-1, 3))[:, 2])

        f0 = self.fun(x0, len(self.key_frame), len(self.ground_truth_ray))
        plt.figure("before residual")
        plt.plot(f0)

        """bound seems not work?"""
        # lower = []
        # upper = []
        # for i in range(len(self.key_frame)):
        #     lower.append(-180)
        #     lower.append(-90)
        #     lower.append(-np.inf)
        #     upper.append(180)
        #     upper.append(90)
        #     upper.append(np.inf)
        #
        # for i in range(len(self.ground_truth_ray)):
        #     lower.append(-180)
        #     lower.append(-90)
        #     upper.append(180)
        #     upper.append(90)

        res = least_squares(self.fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(len(self.key_frame), len(self.ground_truth_ray)))

        res.x[0:3] = self.get_ground_truth_camera(0)

        # for i in range(len(self.key_frame)):

        for i in range(len(self.ground_truth_ray)):
            if res.x[3 *len(self.key_frame) + 2 * i ] > 180 or res.x[3 *len(self.key_frame) + 2 * i ] < -180:
                res.x[3 * len(self.key_frame) + 2 * i] = res.x[3 *len(self.key_frame) + 2 * i ] % 360
            if res.x[3 *len(self.key_frame) + 2 * i + 1] > 90 or res.x[3 *len(self.key_frame) + 2 * i + 1] < -90:
                res.x[3 * len(self.key_frame) + 2 * i + 1] = res.x[3 *len(self.key_frame) + 2 * i +1] % 180 -90

        f1 = self.fun(res.x, len(self.key_frame), len(self.ground_truth_ray))
        plt.figure("after residual")
        plt.plot(f1)

        diff_ray_before = (x0[3 * len(self.key_frame):] - self.ground_truth_x[3 * len(self.key_frame):]).reshape(
            (-1, 2))
        diff_ray_after = (res.x[3 * len(self.key_frame):] - self.ground_truth_x[3 * len(self.key_frame):]).reshape(
            (-1, 2))

        """
        ray difference before and after adjustment
        """
        plt.figure(num="ray error before")
        plt.plot(diff_ray_before[:, 1])

        plt.figure(num="ray error after")
        plt.plot(diff_ray_after[:, 1])

        diff_before = x0 - self.ground_truth_x
        diff_before = diff_before[:3 * len(self.key_frame)].reshape((-1, 3))

        diff_after = res.x - self.ground_truth_x
        diff_after = diff_after[:3 * len(self.key_frame)].reshape((-1, 3))

        """camera difference with ground truth before adjustment"""
        plt.figure(num="before pan")
        plt.plot(diff_before[:, 0])

        plt.figure(num="before tilt")
        plt.plot(diff_before[:, 1])

        plt.figure(num="before f")
        plt.plot(diff_before[:, 2])

        """camera difference with ground truth after adjustment"""
        plt.figure(num="after pan")
        plt.plot(diff_after[:, 0])

        plt.figure(num="after tilt")
        plt.plot(diff_after[:, 1])

        plt.figure(num="after f")
        plt.plot(diff_after[:, 2])

        plt.show()


if __name__ == '__main__':
    bundle_adjust_obj = BundleAdjust("./basketball/basketball_model.mat",
                                     "./basketball/basketball/basketball_anno.mat",
                                     "./synthesize_data_basketball.mat")

    bundle_adjust_obj.bundleAdjust()
