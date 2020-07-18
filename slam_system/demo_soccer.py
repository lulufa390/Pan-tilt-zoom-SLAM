
from sequence_manager import SequenceManager
from ptz_slam import PtzSlam
from util import save_camera_pose

"""
PTZ SLAM experiment code for the soccer sequence.
system parameters:
init_system and add_rays: detect 300 orb keypoints each frame
good new keyframe: 10 ~ 15
images have been blurred. So there is no need to add mask to bounding_box
"""
sequence = SequenceManager("../dataset/soccer/ground_truth.mat",
                            "../dataset/soccer/images",
                            "../dataset/soccer/ground_truth.mat",
                            "../dataset/soccer/player_bounding_box.mat")
slam = PtzSlam()

first_img = sequence.get_image_gray(index=0, dataset_type=1)
first_camera = sequence.get_camera(0)
first_bounding_box = sequence.get_bounding_box_mask(0)

slam.init_system(first_img, first_camera, bounding_box=first_bounding_box)
slam.add_keyframe(first_img, first_camera, 0, enable_rf=False)
# slam.add_keyframe_random_forest(first_img, first_camera, 0)

pan_list = [first_camera.get_ptz()[0]]
tilt_list = [first_camera.get_ptz()[1]]
zoom_list = [first_camera.get_ptz()[2]]

for i in range(1, sequence.length):
    img = sequence.get_image_gray(index=i, dataset_type=1)
    bounding_box = sequence.get_bounding_box_mask(i)
    slam.tracking(next_img=img, bad_tracking_percentage=80, bounding_box=bounding_box)

    if slam.tracking_lost:
        relocalized_camera = slam.relocalize(img, slam.current_camera, enable_rf=False)
        slam.init_system(img, relocalized_camera, bounding_box=bounding_box)

        print("do relocalization!")
    elif slam.new_keyframe:
        slam.add_keyframe(img, slam.current_camera, i, enable_rf=False)
        print("add keyframe!")

    print("=====The ", i, " iteration=====")

    # difference with the ground truth
    dp = slam.cameras[i].pan - sequence.ground_truth_pan[i]
    dt = slam.cameras[i].tilt - sequence.ground_truth_tilt[i]
    df = slam.cameras[i].focal_length - sequence.ground_truth_f[i]
    print('pan, tilt, focal length error: {} {} {}'.format(dp, dt, df))

    pan_list.append(slam.cameras[i].pan)
    tilt_list.append(slam.cameras[i].tilt)
    zoom_list.append(slam.cameras[i].focal_length)

for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
    keyframe.save_to_mat('keyframe_{}.mat'.format(i))

save_camera_pose(pan_list, tilt_list, zoom_list, "./result.mat")