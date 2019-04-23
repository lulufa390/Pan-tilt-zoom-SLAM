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
h_file = './synthetic/homography_vs_ptz/homography-all.mat'
ptz_file = './synthetic/homography_vs_ptz/ptz-all.mat'

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
ptz_gt = np.hstack((pan_gt.reshape(-1, 1), tilt_gt.reshape(-1, 1), fl_gt.reshape(-1, 1)))


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


def mean_std_of_reprojection_error(pts, projected_pts):
    dif = pts - projected_pts
    dif = np.square(dif)
    dif = np.sum(dif, axis=1)
    dif = np.sqrt(dif)
    m, std = np.mean(dif), np.std(dif)
    return (m, std)

def compute_velocity(ptz):
    # approximate angle from pan and tilt
    pan_tilt = ptz[:,0:2]
    pan_tilt = np.square(pan_tilt)
    pan_tilt = np.sum(pan_tilt, axis=1)
    angles = np.sqrt(pan_tilt)
    velocity = np.diff(angles)
    velocity = np.abs(velocity)
    m1, m2 = np.mean(velocity), np.median(velocity)
    return m1, m2

def computer_ground_truth_velocity():
    def compute_velocity(ptz):
        # approximate angle from pan and tilt
        pan_tilt = ptz[:,0:2]
        pan_tilt = np.square(pan_tilt)
        pan_tilt = np.sum(pan_tilt, axis=1)
        angles = np.sqrt(pan_tilt)
        velocity = np.diff(angles)
        velocity = np.abs(velocity)
        m1, m2 = np.mean(velocity), np.median(velocity)
        return m1, m2

    #pan_ptz = data['pan'].squeeze()
    #tilt_ptz = data['tilt'].squeeze()
    #fl_ptz = data['f'].squeeze()
    for i in range(6):
        start_index = i * 600
        end_index = start_index + 600
        cur_ptz = ptz_gt[start_index:end_index]
        m1, m2 = compute_velocity(cur_ptz)
        print('velocity {} {}'.format(m1*60, m2*60))



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
    #sequence_id = [0, 1, 2, 3, 4, 5]
    sequence_id = [2, 3, 1, 0, 5, 4]

    f = plt.figure(figsize=(12, 6))

    for i in range(6):

        #start_index = i * 600
        #end_index = start_index + 600

        plt.subplot(2, 3, i+1)

        start_index = sequence_id[i] * 600
        end_index = start_index + 600
        error1 = error_h[start_index:end_index]
        error2 = error_ptz[start_index:end_index]

        print('reprojeciton error, mean median max :')
        print('homography-based {} {} {}'.format(np.mean(error1), np.median(error1), np.max(error1)))
        print('ours             {} {} {}'.format(np.mean(error2), np.median(error2), np.max(error2)))

        plt.plot(error1, color='b')
        plt.plot(error2, color='r')

        plt.xlim([0, 600])
        plt.ylim([0, 3])
        plt.legend(['Homography-based', 'PTZ (ours)'])
        if i == 3 or i == 4 or i == 5:
            plt.xlabel('Frame number')
        if i == 0 or i == 3:
            plt.ylabel('Reprojection error (pixels)')

    plt.subplots_adjust(hspace=0.25)
    plt.savefig('homography_vs_ptz_synthetic_2.pdf', bbox_inches='tight')
    plt.show()


def compute_relocalization_projection_error():
    import sys
    sys.path
    sys.path.append('../slam_system')
    import copy
    import random
    from sequence_manager import SequenceManager
    sequence = SequenceManager(annotation_path="../../dataset/basketball/synthetic/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")
    base_camera = sequence.camera

    data_folder = './synthetic/relocalization/'
    keyframe_names = ['keyframe-{}.mat'.format(i) for i in range(10, 60, 10)]
    rf_names = ['rf-{}.mat'.format(i) for i in range(10, 60, 10)]

    def load_ptz(file_name):
        data = sio.loadmat(file_name)
        pan = data['pan'].squeeze()
        tilt = data['tilt'].squeeze()
        fl = data['f'].squeeze()

        pan = np.reshape(pan, (-1, 1))
        tilt = np.reshape(tilt, (-1, 1))
        fl = np.reshape(fl, (-1, 1))
        return np.hstack((pan, tilt, fl))

    gt_ptz = load_ptz(data_folder + 'relocalization_gt.mat')
    n = len(keyframe_names)

    def compute_angular_error(gt_ptz, pred_ptz):
        dif = gt_ptz - pred_ptz
        dif = dif[:, 0:2]
        dif = np.square(dif)
        dif = np.sum(dif, axis=1)
        errors = np.sqrt(dif)
        return errors


    threshold = 2.0

    for i in range(n):

        # for each outlier level
        keyframe_ptz = load_ptz(data_folder + keyframe_names[i])
        rf_ptz = load_ptz(data_folder + rf_names[i])


        num_camera = gt_ptz.shape[0]

        keyframe_angular_errors = compute_angular_error(gt_ptz, keyframe_ptz)
        rf_angular_errors = compute_angular_error(gt_ptz, rf_ptz)
        #print(np.where(keyframe_angular_errors < threshold)[0].shape[0])
        p1 = np.where(keyframe_angular_errors < threshold)[0].shape[0] /num_camera
        p2 = np.where(rf_angular_errors < threshold)[0].shape[0] /num_camera


        camera_gt = copy.deepcopy(base_camera)
        camera_keyframe = copy.deepcopy(base_camera)
        camera_rf = copy.deepcopy(base_camera)

        # sample rays from image space
        im_w, im_h = 1280, 720
        point_num = 50


        keyframe_reprojection_error = np.zeros(num_camera)
        ours_reprojection_error = np.zeros(num_camera)
        for j in range(num_camera):
            cur_gt_ptz = gt_ptz[j]
            cur_keyframe_ptz = keyframe_ptz[j]
            cur_rf_ptz = rf_ptz[j]
            #print('PTZ parameter: {} {} {}'.format(cur_gt_ptz, cur_keyframe_ptz, cur_rf_ptz))

            camera_gt.set_ptz((cur_gt_ptz[0], cur_gt_ptz[1], cur_gt_ptz[2]))
            camera_keyframe.set_ptz((cur_keyframe_ptz[0], cur_keyframe_ptz[1], cur_keyframe_ptz[2]))
            camera_rf.set_ptz((cur_rf_ptz[0], cur_rf_ptz[1], cur_rf_ptz[2]))

            points = np.zeros((point_num, 2))
            for k in range(point_num):
                points[k][0] = random.randint(0, im_w - 1)
                points[k][1] = random.randint(0, im_h - 1)

            rays = camera_gt.back_project_to_rays(points)

            points_keyframe, _ = camera_keyframe.project_rays(rays)
            points_rf, _ = camera_rf.project_rays(rays)

            m1, std1 = mean_std_of_reprojection_error(points, points_keyframe)
            m2, std2 = mean_std_of_reprojection_error(points, points_rf)
            keyframe_reprojection_error[j] = m1
            ours_reprojection_error[j] = m2

        print('outlier percentage: {}'.format((i+1)*10))
        print('mean: keyframe: {}, ours: {}'.format(np.mean(keyframe_reprojection_error), np.mean(ours_reprojection_error)))
        print('std: keyframe: {}, ours: {}'.format(np.std(keyframe_reprojection_error),
                                                    np.std(ours_reprojection_error)))
        print('correct relocalization: keyframe: {}, ours: {}'.format(p1, p2))


def plot_gt_ptz():
    f = plt.figure()

    plt.plot(pan_ptz)
    #plt.plot(tilt_gt)
    plt.xlabel('Frame numbers')
    plt.ylabel('Pan angle (degrees)')
    plt.xlim([0,3600])
    plt.savefig('pan_pred.pdf', bbox_inches='tight')
    plt.show()

def vis_rf_keyframe():
    import PIL.Image as Image
    data = sio.loadmat('/Users/jimmy/Desktop/basketball_standard_rf/keyframes/682.mat')
    print(data.keys())


    im_name = '/Users/jimmy/Code/ptz_slam/dataset/basketball/seq1/images/00084682.jpg'
    im = Image.open(im_name)

    points = data['keypoint']
    print(points.shape)

    f = plt.figure()
    plt.imshow(im)
    plt.plot(points[:, 0], points[:, 1], '*')
    plt.show()


#rmse()
#trajectory()
#reprojection_error()
#vis_reprojection_error_area()
#vis_reprojection_error_multiple_sequence()
#vis_reprojection_error_multiple_sequence_subplot()

#computer_ground_truth_velocity()

#compute_relocalization_projection_error()
#plot_gt_ptz()

vis_rf_keyframe()


"""
print(pan_gt.shape)
f = plt.figure()
plt.plot(pan_gt, '-')
plt.plot(pan_h, '-')
plt.plot(pan_ptz, '-')
plt.legend(['ground truth', 'homography', 'PTZ'])
plt.show()
"""


