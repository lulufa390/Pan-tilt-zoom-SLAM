from sequence_manager import SequenceManager
from util import load_camera_pose
from image_process import detect_sift
from util import add_gauss

import cv2 as cv
import numpy as np
import math
import copy
import random

def ut_single_image():
    sequence = SequenceManager(annotation_path="../../dataset/basketball/synthetic/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")

    gt_pan, gt_tilt, gt_f = load_camera_pose("../../dataset/synthesized/synthesize_ground_truth.mat", separate=True)

    # read image and ground truth pose
    im = cv.imread('/Users/jimmy/Code/ptz_slam/dataset/basketball/synthetic/images/00000000.jpg', 0)
    pan, tilt, fl = gt_pan[0], gt_tilt[0], gt_f[0]
    gt_pose = [pan, tilt, fl]
    camera = sequence.camera
    camera.set_ptz((pan, tilt, fl))

    im_w, im_h = 1280, 720
    points = detect_sift(im, 20)
    N = points.shape[0]
    print(points.shape)

    rays = camera.back_project_to_rays(points)
    print(rays.shape)

    from relocalization import _compute_residual
    from scipy.optimize import least_squares
    from transformation import TransFunction

    def robust_test(variance, camera):
        """
        return mean value of reprojection error
        :param variance:
        :param camera:
        :return:
        """
        # add noise to pts
        noise_pts = add_gauss(points, variance, im_w, im_h)

        # add noise to camera pose
        init_pose = np.zeros(3)
        init_pose[0] = pan + np.random.normal(0, 5.0)
        init_pose[1] = tilt + np.random.normal(0, 2.0)
        init_pose[2] = fl + np.random.normal(0, 150)

        # optimized the camera pose
        optimized_pose = least_squares(_compute_residual, init_pose, verbose=0, x_scale='jac', ftol=1e-4, method='trf',
                                       args=(rays, noise_pts, im_w / 2, im_h / 2))
        optimzied_ptz = optimized_pose.x
        #print('ground truth: {}'.format(gt_pose))
        #print('estiamted pose: {}'.format(optimzied_ptz))

        # compute reprojection error
        estimated_camera = copy.deepcopy(camera)
        estimated_camera.set_ptz(optimzied_ptz)

        pts1, _ = camera.project_rays(rays)
        pts2, _ = estimated_camera.project_rays(rays)

        reprojection_error = pts1 - pts2
        for i in range(N):
            dx, dy = pts1[i] - pts2[i]
            reprojection_error[i] = math.sqrt(dx * dx + dy * dy)
        # print(reprojection_error[0:10])
        m, std = np.mean(reprojection_error), np.std(reprojection_error)
        #print('mean std: {} {}'.format(m, std))
        return m

    variances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    repeat_num = 20

    for v in variances:
        error_mean = np.zeros(repeat_num)
        # repeat
        for i in range(repeat_num):
            m = robust_test(v, camera)
            error_mean[i] = m
        m, std = np.mean(error_mean), np.std(error_mean)
        print('noise, mean, std: {} {} {}'.format(v, m, std))













if __name__ == "__main__":
    ut_single_image()