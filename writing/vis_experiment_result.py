import numpy as np
import scipy.io as sio
import matplotlib
from sys import platform as sys_pf

if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
elif sys_pf == 'win32':
    matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

data = sio.loadmat('../../dataset/synthesized/synthesize_ground_truth.mat')
pan_gt = data['pan'].squeeze()
tilt_gt = data['tilt'].squeeze()
fl_gt = data['f'].squeeze()

data = sio.loadmat('./homography_vs_ptz_EKF/homography_ekf.mat')
pan_h = data['pan'].squeeze()
tilt_h = data['tilt'].squeeze()
fl_h = data['f'].squeeze()

data = sio.loadmat('./homography_vs_ptz_EKF/ptz_ekf_tracking.mat')
pan_ptz = data['pan'].squeeze()
tilt_ptz = data['tilt'].squeeze()
fl_ptz = data['f'].squeeze()


def rmse():
    from sklearn.metrics import mean_squared_error
    error1 = mean_squared_error(pan_gt, pan_h)
    error2 = mean_squared_error(pan_gt, pan_ptz)
    print('{} {}'.format(error1, error2))

    error1 = mean_squared_error(fl_gt, fl_h)
    error2 = mean_squared_error(fl_gt, fl_ptz)
    print('{} {}'.format(error1, error2))

def trajectory():
    plt.plot(fl_gt)
    plt.plot(fl_h)
    plt.plot(fl_ptz)
    plt.legend(['ground truth', 'homography', 'PTZ'])
    plt.show()

def reprojection_error():
    import sys
    sys.path
    sys.path.append('../slam_system')
    import copy
    import random
    from sequence_manager import SequenceManager
    sequence = SequenceManager(annotation_path="../../dataset/basketball/synthetic/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")
    camera = sequence.camera
    #camera.set_ptz((pan, tilt, fl))
    n = pan_gt.shape[0]
    reprojection_error_ptz_mean_std = np.zeros((n, 2))
    reprojection_error_h_mean_std = np.zeros((n, 2))
    for i in range(n):
        # randomly generate keypoints
        camera_gt = copy.deepcopy(camera)
        camera_h = copy.deepcopy(camera)
        camera_ptz = copy.deepcopy(camera)

        camera_gt.set_ptz((pan_gt[i], tilt_gt[i], fl_gt[i]))
        camera_h.set_ptz((pan_h[i], tilt_h[i], fl_h[i]))
        camera_ptz.set_ptz((pan_ptz[i], tilt_ptz[i], fl_ptz[i]))

        im_w, im_h = 1280, 720
        point_num = 50
        points = np.zeros((point_num, 2))
        for j in range(point_num):
            points[j][0] = random.randint(0, im_w-1)
            points[j][1] = random.randint(0, im_h-1)

        rays = camera_gt.back_project_to_rays(points)

        points_h, _ = camera_h.project_rays(rays)
        points_ptz, _ = camera_ptz.project_rays(rays)

        def mean_std_of_reprojection_error(pts, projected_pts):
            dif = pts - projected_pts
            dif = np.square(dif)
            dif = np.sum(dif, axis=1)
            dif = np.sqrt(dif)
            m, std = np.mean(dif), np.std(dif)
            return (m, std)
        m1, std1 = mean_std_of_reprojection_error(points, points_h)
        m2, std2 = mean_std_of_reprojection_error(points, points_ptz)
        #print('{} {} {} {}'.format(m1, std1, m2, std2))
        reprojection_error_h_mean_std[i] = m1, std1
        reprojection_error_ptz_mean_std[i] = m2, std2

    # save this file
    sio.savemat('homography_ptz_reprojection_error.mat', {'reprojection_error_h_mean_std':reprojection_error_h_mean_std,
                                                          'reprojection_error_ptz_mean_std':reprojection_error_ptz_mean_std})
    plt.plot(reprojection_error_h_mean_std[:, 0])
    plt.plot(reprojection_error_ptz_mean_std[:,0])
    plt.legend(['homography-based', 'PTZ (ours)'])
    plt.show()

def vis_reprojection_error():
    data = sio.loadmat('homography_ptz_reprojection_error.mat')
    error_h = data['reprojection_error_h_mean_std']
    error_ptz = data['reprojection_error_ptz_mean_std']

    plt.plot(error_ptz[:,0])
    plt.show()


#rmse()
#trajectory()
reprojection_error()
#vis_reprojection_error()


"""
print(pan_gt.shape)
f = plt.figure()
plt.plot(pan_gt, '-')
plt.plot(pan_h, '-')
plt.plot(pan_ptz, '-')
plt.legend(['ground truth', 'homography', 'PTZ'])
plt.show()
"""


