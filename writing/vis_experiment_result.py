import numpy as np
import scipy.io as sio
import matplotlib
from sys import platform as sys_pf

if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
elif sys_pf == 'win32':
    matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

gt_file = '../../dataset/synthesized/synthesize_ground_truth.mat'
h_file = './gt_6_sequence/merged_estimation/homography-all.mat'
ptz_file = './gt_6_sequence/merged_estimation/ptz-all.mat'

#gt_file = '../../dataset/synthesized/synthesize_ground_truth.mat'
#h_file = './homography_vs_ptz_EKF/homography_ekf.mat'
#ptz_file = './homography_vs_ptz_EKF/ptz_ekf_tracking.mat'

#gt_file = './gt_6_sequence/0-600.mat'
#h_file = './gt_6_sequence/estimation/homography-0.mat'
#ptz_file = './gt_6_sequence/estimation/ptz-0.mat'

data = sio.loadmat(gt_file)
pan_gt = data['pan'].squeeze()
tilt_gt = data['tilt'].squeeze()
fl_gt = data['f'].squeeze()

data = sio.loadmat(h_file)
pan_h = data['pan'].squeeze()
tilt_h = data['tilt'].squeeze()
fl_h = data['f'].squeeze()

data = sio.loadmat(ptz_file)
pan_ptz = data['pan'].squeeze()
tilt_ptz = data['tilt'].squeeze()
fl_ptz = data['f'].squeeze()

def load_data():
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
        point_num = 100
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

def vis_reprojection_error_area():
    data = sio.loadmat('homography_ptz_reprojection_error.mat')
    error_h = data['reprojection_error_h_mean_std'][0:600,:]
    error_ptz = data['reprojection_error_ptz_mean_std'][0:600,:]

    error_upper = error_h[:,0] + error_h[:,1]
    error_lower = error_h[:,0] - error_h[:,1]


    x = error_h[:,0].tolist()
    y1 = error_lower.tolist()
    y2 = error_upper.tolist()
    print('{} {} {}'.format(len(x), len(y1), len(y2)))

    print('{} {} {}'.format(len(x), len(y1), len(y2)))
    # Shade the area between y1 and y2

    plt.fill_between(range(600), y1, y2,
                     facecolor="orange",  # The fill color
                     color='blue',  # The outline color
                     alpha=0.2)  # Transparency of the fill
    plt.plot(x, '-')
    plt.show()

def vis_reprojection_error_multiple_sequence():
    data = sio.loadmat('homography_ptz_reprojection_error.mat')
    error_h = data['reprojection_error_h_mean_std'][:, 0]
    error_ptz = data['reprojection_error_ptz_mean_std'][:, 0]

    print(error_h.shape)
    sequence_id = [0, 3,  5]
    f = plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(len(sequence_id)):
        start_index = sequence_id[i] * 600
        end_index = start_index + 600
        error1 = error_h[start_index:end_index]
        error2 = error_ptz[start_index:end_index]

        plt.plot(error1, '-', color=colors[i])
        plt.plot(error2, '-.', color=colors[i], dashes=(5, 10))

    plt.legend(['H-s1', 'PTZ-s1 (ours)','H-s2', 'PTZ-s2 (ours)','H-s3', 'PTZ-s3 (ours)'])
    plt.xlim([0, 600])
    plt.ylim([0, 10])
    plt.xlabel('Frame number')
    plt.ylabel('Reprojection error (pixels)')
    plt.show()

def vis_reprojection_error_multiple_sequence_subplot():
    data = sio.loadmat('homography_ptz_reprojection_error.mat')
    error_h = data['reprojection_error_h_mean_std'][:, 0]
    error_ptz = data['reprojection_error_ptz_mean_std'][:, 0]

    print(error_h.shape)
    sequence_id = [0, 1, 2, 3, 4, 5]
    f = plt.figure(figsize=(12, 6))

    for i in range(6):
        plt.subplot(2, 3, i+1)

        start_index = sequence_id[i] * 600
        end_index = start_index + 600
        error1 = error_h[start_index:end_index]
        error2 = error_ptz[start_index:end_index]

        plt.plot(error1, color='b')
        plt.plot(error2, color='r')

        plt.xlim([0, 600])
        plt.ylim([0, 3])
        plt.legend(['Homography-based', 'PTZ (ours)'])
        plt.xlabel('Frame number')
        if i == 0 or i == 3:
            plt.ylabel('Reprojection error (pixels)')

    plt.subplots_adjust(hspace=0.25)
    plt.savefig('homography_vs_ptz_synthetic.pdf', bbox_inches='tight')
    plt.show()


#rmse()
#trajectory()
#reprojection_error()
#vis_reprojection_error_area()
#vis_reprojection_error_multiple_sequence()
vis_reprojection_error_multiple_sequence_subplot()


"""
print(pan_gt.shape)
f = plt.figure()
plt.plot(pan_gt, '-')
plt.plot(pan_h, '-')
plt.plot(pan_ptz, '-')
plt.legend(['ground truth', 'homography', 'PTZ'])
plt.show()
"""


