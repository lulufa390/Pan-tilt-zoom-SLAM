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
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, model_path, annotation_path, data_path):
        """
        :param model_path: path for basketball court model
        :param annotation_path: path for camera sequence information
        :param data_path: path for synthesized points
        """

        # random.seed(1)
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

        data = sio.loadmat(data_path)

        """ this is synthesized rays to generate 2d-point. Real data does not have this variable """
        self.ground_truth_ray = data["rays"]

        """
        initialize the fixed parameters of our algorithm
        u, v, base_rotation and c
        """
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        self.key_frame = np.array([i for i in range(0, 3600, 100)])

        """
        parameters to be updated
        """
        self.ground_truth_pan = np.ndarray([len(self.key_frame)])
        self.ground_truth_tilt = np.ndarray([len(self.key_frame)])
        self.ground_truth_f = np.ndarray([len(self.key_frame)])
        for i in range(len(self.key_frame)):
            self.ground_truth_pan[i], self.ground_truth_tilt[i], self.ground_truth_f[i] \
                = self.annotation[0][self.key_frame[i]]['ptz'].squeeze()

        """ground_truth_x is a 1d array for ground truth camera pose and rays"""
        self.ground_truth_x = np.ndarray([3 * len(self.key_frame) + 2 * len(self.ground_truth_ray)])

        """camera and rays with noise"""
        self.cameraArray = np.ndarray([3 * len(self.key_frame)])
        for i in range(len(self.key_frame)):
            self.cameraArray[3 * i] = self.ground_truth_pan[i] + random.gauss(0, 5)
            self.cameraArray[3 * i + 1] = self.ground_truth_tilt[i] + random.gauss(0, 2)
            self.cameraArray[3 * i + 2] = self.ground_truth_f[i] + random.gauss(0, 20)

            self.ground_truth_x[3 * i] = self.ground_truth_pan[i]
            self.ground_truth_x[3 * i + 1] = self.ground_truth_tilt[i]
            self.ground_truth_x[3 * i + 2] = self.ground_truth_f[i]

        self.ray3d = np.ndarray([2 * len(self.ground_truth_ray)])
        for i in range(len(self.ground_truth_ray)):
            self.ray3d[2 * i] = self.ground_truth_ray[i][0] + random.gauss(0, 5)
            self.ray3d[2 * i + 1] = self.ground_truth_ray[i][1] + random.gauss(0, 2)

            self.ground_truth_x[3 * len(self.key_frame) + 2 * i] = self.ground_truth_ray[i][0]
            self.ground_truth_x[3 * len(self.key_frame) + 2 * i + 1] = self.ground_truth_ray[i][1]


        self.image_list = []
        self.point_index_list = []
        # random.seed(1)
        for i in range(len(self.key_frame)):
            tmp_image, tmp_index = self.generate_image(i)
            self.image_list.append(tmp_image)
            self.point_index_list.append(tmp_index)


    def get_ground_truth_camera(self, index):
        return np.array([self.ground_truth_pan[index], self.ground_truth_tilt[index], self.ground_truth_f[index]])

    def generate_image(self, index):
        """
        :param index: index for image
        :return: all points in that observation
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
        return self.image_list[i], self.point_index_list[i]

    def get_observation_from_rays(self, pan, tilt, f, rays, ray_index):
        points = np.ndarray([0, 2])

        for j in range(len(ray_index)):
            theta = rays[int(ray_index[j])][0]
            phi = rays[int(ray_index[j])][1]
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, theta, phi)
            # if 0 < tmp[0] < self.width and 0 < tmp[1] < self.height:
            points = np.row_stack([points, np.asarray(tmp)])

        return points

    def fun(self, params, n_cameras, n_points):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
        points_3d = params[n_cameras * 3:].reshape((n_points, 2))


        residual = np.ndarray([0])

        for i in range(n_cameras):
            point_2d, inner_index = self.get_image(i)

            # print(len(point_2d), len(inner_index))

            proj_point = self.get_observation_from_rays(
                camera_params[i, 0], camera_params[i, 1], camera_params[i, 2], points_3d, inner_index)

            residual = np.append(residual, proj_point.ravel() - point_2d.ravel())

        # print(np.linalg.norm(residual))

        # print(residual.shape)

        return residual

    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """

        x0 = np.hstack((self.cameraArray, self.ray3d))


        plt.figure(num="ground pan")
        plt.plot(self.ground_truth_x[:3 * len(self.key_frame)].reshape((-1,3))[:, 0])

        plt.figure(num="ground tilt")
        plt.plot(self.ground_truth_x[:3 * len(self.key_frame)].reshape((-1,3))[:, 1])

        plt.figure(num="ground f")
        plt.plot(self.ground_truth_x[:3 * len(self.key_frame)].reshape((-1,3))[:, 2])


        # f0 = self.fun(x0, len(self.key_frame), len(self.ground_truth_ray))
        #
        # plt.figure("before residual")
        # plt.plot(f0)


        res = least_squares(self.fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(len(self.key_frame), len(self.ground_truth_ray)))


        # f1 = self.fun(res.x, len(self.key_frame), len(self.ground_truth_ray))
        # plt.figure("after residual")
        # plt.plot(f1)


        print(np.linalg.norm(x0 - self.ground_truth_x))
        print(np.linalg.norm(res.x - self.ground_truth_x))
        # params = self.optimizedParams(res.x, len(self.key_frame), len(self.ground_truth_ray))

        diff_before = x0 - self.ground_truth_x
        diff_before = diff_before[:3 * len(self.key_frame)].reshape((-1,3))

        diff_after = res.x - self.ground_truth_x
        diff_after = diff_after[:3 * len(self.key_frame)].reshape((-1, 3))

        after_camera = res.x[:3 * len(self.key_frame)].reshape((-1,3))

        # plt.figure(num="after camera pan")
        # plt.plot(after_camera[:, 0])
        #
        # plt.figure(num="after camera tilt")
        # plt.plot(after_camera[:, 1])
        #
        # plt.figure(num="after camera f")
        # plt.plot(after_camera[:, 2])


        plt.figure(num="before pan")
        plt.plot(diff_before[:, 0])

        plt.figure(num="before tilt")
        plt.plot(diff_before[:, 1])

        plt.figure(num="before f")
        plt.plot(diff_before[:, 2])

        plt.figure(num="after pan")
        plt.plot(diff_after[:, 0])

        plt.figure(num="after tilt")
        plt.plot(diff_after[:, 1])

        plt.figure(num="after f")
        plt.plot(diff_after[:, 2])


        plt.show()



        # return params


if __name__ == '__main__':
    bundle_adjust_obj = BundleAdjust("./basketball/basketball_model.mat",
                                     "./basketball/basketball/basketball_anno.mat",
                                     "./synthesize_data_basketball.mat")

    bundle_adjust_obj.bundleAdjust()
