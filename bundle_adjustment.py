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

        # self.ground_truth_pan = sig.savgol_filter(self.ground_truth_pan, 181, 1)
        # self.ground_truth_tilt = sig.savgol_filter(self.ground_truth_tilt, 181, 1)
        # self.ground_truth_f = sig.savgol_filter(self.ground_truth_f, 181, 1)

        """camera and rays with noise"""
        self.cameraArray = np.ndarray([3 * len(self.key_frame)])
        for i in range(len(self.key_frame)):
            self.cameraArray[3 * i] = self.ground_truth_pan[i] + random.gauss(0, 0.5)
            self.cameraArray[3 * i + 1] = self.ground_truth_tilt[i] + random.gauss(0, 0.5)
            self.cameraArray[3 * i + 2] = self.ground_truth_f[i] + random.gauss(0, 50)

        self.points3D = np.ndarray([2 * len(self.ground_truth_ray)])
        for i in range(len(self.ground_truth_ray)):
            self.points3D[2 * i] = self.ground_truth_ray[i][0] + random.gauss(0, 0.5)
            self.points3D[2 * i + 1] = self.ground_truth_ray[i][1] + random.gauss(0, 0.5)

    def get_ground_truth_camera(self, index):
        return np.array([self.ground_truth_pan[index], self.ground_truth_tilt[index], self.ground_truth_f[index]])

    def get_image(self, index, inner_idx):
        """
        :param index: index for image
        :return: all points in that observation
        """
        pan, tilt, f = self.get_ground_truth_camera(index)
        points = np.ndarray([0, self.ground_truth_ray.shape[1]])

        for i in range(len(inner_idx)):
            j = int(inner_idx[i])

            x, y = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, self.ground_truth_ray[j][0],
                                                     self.ground_truth_ray[j][1])
            # x += random.gauss(0, 2)
            # y += random.gauss(0, 2)

            # if 0 < x < self.width and 0 < y < self.height:
            points = np.row_stack([points, np.array([x, y])])

        return points

    def get_observation_from_rays(self, pan, tilt, f, rays):
        points = np.ndarray([0, 2])
        index = np.ndarray([0])

        for j in range(len(rays)):
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, rays[j][0], rays[j][1])
            if 0 < tmp[0] < self.width and 0 < tmp[1] < self.height:
                points = np.row_stack([points, np.asarray(tmp)])
                index = np.concatenate([index, [j]], axis=0)

        return points, index

    def fun(self, params, n_cameras, n_points):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
        points_3d = params[n_cameras * 3:].reshape((n_points, 2))



        residual = np.ndarray([0])

        for i in range(n_cameras):
            proj_point, inner_index = self.get_observation_from_rays(
                camera_params[i, 0], camera_params[i, 1], camera_params[i, 2], points_3d)

            point_2d = self.get_image(i, inner_index)

            residual = np.append(residual, proj_point.ravel() - point_2d.ravel())

        # print(np.linalg.norm(residual))



        # print(residual.shape)

        return residual

    def bundle_adjustment_sparsity(self, numCameras, numPoints):
        m = cameraIndices.size * 2
        n = numCameras * 9 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(9):
            A[2 * i, cameraIndices * 9 + s] = 1
            A[2 * i + 1, cameraIndices * 9 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 9 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 9 + pointIndices * 3 + s] = 1

        return A

    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
        points_3d = params[n_cameras * 3:].reshape((n_points, 2))

        return camera_params, points_3d

    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """

        x0 = np.hstack((self.cameraArray, self.points3D))

        f0 = self.fun(x0, len(self.key_frame), len(self.ground_truth_ray))

        plt.figure(0)
        plt.plot(f0)

        # A = self.bundle_adjustment_sparsity(
        #     self.annotation.size, len(self.ground_truth_ray))

        res = least_squares(self.fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(len(self.key_frame), len(self.ground_truth_ray)))

        f1 = self.fun(res.x, len(self.key_frame), len(self.ground_truth_ray))
        plt.figure(1)
        plt.plot(f1)
        plt.show()

        params = self.optimizedParams(res.x, len(self.key_frame), len(self.ground_truth_ray))

        return params


if __name__ == '__main__':
    bundle_adjust_obj = BundleAdjust("./basketball/basketball_model.mat",
                                     "./basketball/basketball/basketball_anno.mat",
                                     "./synthesize_data_basketball.mat")

    bundle_adjust_obj.bundleAdjust()
