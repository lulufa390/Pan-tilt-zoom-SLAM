from ptz_slam import *


def ut_soccer3():
    """
    PTZ SLAM experiment code for soccer sequence3.
    system parameters:
    init_system and add_rays: detect 300 orb keypoints each frame
    good new keyframe: 10 ~ 15
    images have been blurred. So there is no need to add mask to bounding_box
    """
    sequence = SequenceManager("../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_330",
                               "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_player_bounding_box.mat")
    slam = PtzSlam()

    first_img = sequence.get_image_gray(index=0, dataset_type=1)
    first_camera = sequence.get_camera(0)
    first_bounding_box = sequence.get_bounding_box_mask(0)

    slam.init_system(first_img, first_camera, first_bounding_box)
    slam.add_keyframe(first_img, first_camera, 0)

    for i in range(1, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=1)
        bounding_box = sequence.get_bounding_box_mask(i)
        slam.tracking(next_img=img, bad_tracking_percentage=80, bounding_box=bounding_box)

        if slam.tracking_lost:
            relocalized_camera = slam.relocalize(img, slam.current_camera)
            slam.init_system(img, relocalized_camera, bounding_box)

            print("do relocalization!")
        elif slam.new_keyframe:
            slam.add_keyframe(img, slam.current_camera, i)
            print("add keyframe!")

        print("=====The ", i, " iteration=====")

        print("%f" % (slam.cameras[i].pan - sequence.ground_truth_pan[i]))
        print("%f" % (slam.cameras[i].tilt - sequence.ground_truth_tilt[i]))
        print("%f" % (slam.cameras[i].focal_length - sequence.ground_truth_f[i]))


def ut_basketball():
    """
    PTZ SLAM experiment code for basketball dataset.
    system parameters:
    init_system and add_rays: detect 300 orb keypoints each frame
    good new keyframe: 10 ~ 25
    """

    sequence = SequenceManager("../../dataset/basketball/ground_truth.mat",
                               "../../dataset/basketball/images",
                               "../../dataset/basketball/ground_truth.mat",
                               "../../dataset/basketball/bounding_box.mat")
    slam = PtzSlam()

    first_img = sequence.get_image_gray(index=0, dataset_type=0)
    first_camera = sequence.get_camera(0)
    first_bounding_box = sequence.get_bounding_box_mask(0)

    slam.init_system(first_img, first_camera, first_bounding_box)
    slam.add_keyframe(first_img, first_camera, 0)

    for i in range(1, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=0)
        bounding_box = sequence.get_bounding_box_mask(i)
        slam.tracking(next_img=img, bad_tracking_percentage=80, bounding_box=bounding_box)

        if slam.tracking_lost:
            relocalized_camera = slam.relocalize(img, slam.current_camera)
            slam.init_system(img, relocalized_camera, bounding_box)

            print("do relocalization!")
        elif slam.new_keyframe:
            slam.add_keyframe(img, slam.current_camera, i)
            print("add keyframe!")

        print("=====The ", i, " iteration=====")

        print("%f" % (slam.cameras[i].pan - sequence.ground_truth_pan[i]))
        print("%f" % (slam.cameras[i].tilt - sequence.ground_truth_tilt[i]))
        print("%f" % (slam.cameras[i].focal_length - sequence.ground_truth_f[i]))

        slam.keyframe_map.save_keyframes_to_mat("../../map/map_data.mat")

        # for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
        #     keyframe.save_to_mat("../../map/" + str(i) + ".mat")

    slam.keyframe_map.save_keyframes_to_mat("../../map/map_data.mat")

    # for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
    #     keyframe.save_to_mat("../../map/" + str(i) + ".mat")


# deprecated Olympics hockey dataset
# def ut_hockey():
#     slam = PtzSlam()
#     sequence_length = 26
#
#     annotation = sio.loadmat("../../ice_hockey_1/olympic_2010_reference_frame.mat")
#
#     filename = annotation["images"]
#     ptzs = annotation["opt_ptzs"]
#     cameras = annotation["opt_cameras"]
#     shared_parameters = annotation["shared_parameters"]
#
#     first_img_color = cv.imread("../../ice_hockey_1/olympic_2010_reference_frame/image/" + filename[0])
#     first_img_gray = cv.cvtColor(first_img_color, cv.COLOR_BGR2GRAY)
#
#     first_camera = PTZCamera(cameras[0, 0:2], shared_parameters[0:3, 0],
#                              shared_parameters[3:6, 0], shared_parameters[6:12, 0])
#     first_camera.set_ptz(ptzs[0])
#
#     slam.init_system(first_img_gray, first_camera)
#     slam.add_keyframe(first_img_gray, first_camera, 0)
#
#     for i in range(1, sequence_length):
#         next_img_color = cv.imread("../../ice_hockey_1/olympic_2010_reference_frame/image/" + filename[i])
#         next_img_gray = cv.cvtColor(next_img_color, cv.COLOR_BGR2GRAY)
#
#         slam.tracking(next_img=next_img_gray, bad_tracking_percentage=80, bounding_box=None)
#
#         if slam.tracking_lost:
#             relocalized_camera = slam.relocalize(next_img_gray, slam.current_camera)
#             slam.init_system(next_img_gray, relocalized_camera)
#
#             print("do relocalization!")
#         elif slam.new_keyframe:
#             slam.add_keyframe(next_img_gray, slam.current_camera, i)
#             print("add keyframe!")
#
#         print("=====The ", i, " iteration=====")
#         print("%f" % (slam.cameras[i].pan - ptzs[i, 0]))
#         print("%f" % (slam.cameras[i].tilt - ptzs[i, 1]))
#         print("%f" % (slam.cameras[i].focal_length - ptzs[i, 2]))


def ut_UBC_hockey():
    """
    experiment code for UBC hockey data
    system parameters:
    init_system and add_rays: detect 500 sift keypoints each frame
    good new keyframe: ???
    """
    slam = PtzSlam()
    sequence_length = 900

    annotation_mat = sio.loadmat("../../UBC_2017/UBC_2017/UBC_hockey_ground_truth.mat")
    bounding_box_mat_path = "../../UBC_2017/UBC_2017/bounding_box.mat"

    filename_list = ["000" + str(i + 48630) + ".jpg" for i in range(sequence_length)]
    camera_list = annotation_mat["camera"]

    ptz_list = annotation_mat["ptz"]
    image_name_list = annotation_mat["image_name"].squeeze()

    ptz_extend_list = -1 * np.ones([sequence_length, 3])

    for i, image_name in enumerate(image_name_list, 0):
        index = int(image_name[0][0:-4]) - 48630
        ptz_extend_list[index, 0:3] = ptz_list[i, 0:3]

    camera_center = annotation_mat["cc"].squeeze()
    base_rotation = annotation_mat["base_rotation"].squeeze()

    bounding_box_manager = SequenceManager(bounding_box_path=bounding_box_mat_path)

    first_img = cv.imread("../../UBC_2017/UBC_2017/images/" + filename_list[0],
                          cv.IMREAD_GRAYSCALE)

    first_camera = PTZCamera(camera_list[0, 0:2], camera_center, base_rotation)

    first_camera.set_ptz(ptz_extend_list[0])

    slam.init_system(first_img, first_camera, bounding_box_manager.get_bounding_box_mask(30, 0.5))
    slam.add_keyframe(first_img, first_camera, 0)

    with open("result.txt", 'w+') as f:
        for i in range(1, sequence_length):
            next_img = cv.imread("../../UBC_2017/UBC_2017/images/" + filename_list[i],
                                 cv.IMREAD_GRAYSCALE)

            slam.tracking(next_img=next_img, bad_tracking_percentage=40,
                          bounding_box=bounding_box_manager.get_bounding_box_mask(i + 30, 0.5))

            if slam.tracking_lost:
                relocalized_camera = slam.relocalize(next_img, slam.current_camera)
                slam.init_system(next_img, relocalized_camera, bounding_box_manager.get_bounding_box_mask(30 + i, 0.5))
                print("do relocalization!")

            elif slam.new_keyframe:
                slam.add_keyframe(next_img, slam.current_camera, i)
                print("add keyframe!")

            print("=====The ", i, " iteration=====")

            print("Pan: %f" % slam.cameras[i].pan)
            print("Tilt: %f" % slam.cameras[i].tilt)
            print("Zoom: %f" % slam.cameras[i].focal_length)

            if not np.all(ptz_extend_list[i] == np.array([-1, -1, -1])):
                print("Pan Error: %f" % (slam.cameras[i].pan - ptz_extend_list[i, 0]))
                print("Tilt Error: %f" % (slam.cameras[i].tilt - ptz_extend_list[i, 1]))
                print("zoom Error: %f" % (slam.cameras[i].focal_length - ptz_extend_list[i, 2]))

                f.write(str(i) + "th frame\n")
                f.write("Pan Error: %f\n" % (slam.cameras[i].pan - ptz_extend_list[i, 0]))
                f.write("Tilt Error: %f\n" % (slam.cameras[i].tilt - ptz_extend_list[i, 1]))
                f.write("zoom Error: %f\n" % (slam.cameras[i].focal_length - ptz_extend_list[i, 2]))


def baseline_keyframe_based_homography_matching_basketball():
    sequence = SequenceManager("../../dataset/basketball/ground_truth.mat",
                               "../../dataset/basketball/images",
                               "../../dataset/basketball/ground_truth.mat",
                               "../../dataset/basketball/bounding_box.mat")

    keyframes = Map('sift')

    # keyframes_list = [0, 237, 664, 683, 700, 722, 740, 778, 2461]
    keyframes_list = [0, 650, 698, 730, 804]

    pan_list = []
    tilt_list = []
    zoom_list = []

    for i in keyframes_list:
        img = sequence.get_image_gray(i)
        p, t, z = sequence.ground_truth_pan[i], sequence.ground_truth_tilt[i], sequence.ground_truth_f[i]

        u, v = sequence.camera.principal_point
        c = sequence.camera.camera_center
        r = sequence.camera.base_rotation

        new_keyframe = KeyFrame(img, i, c, r, u, v, p, t, z)

        keyframes.add_keyframe_without_ba(new_keyframe)

    for i in range(0, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=0)
        lost_pose = 0, 0, 3000

        relocalize_pose = relocalization_camera(keyframes, img, lost_pose)

        print("=====The ", i, " iteration=====")

        print("%f" % (relocalize_pose[0] - sequence.ground_truth_pan[i]))
        print("%f" % (relocalize_pose[1] - sequence.ground_truth_tilt[i]))
        print("%f" % (relocalize_pose[2] - sequence.ground_truth_f[i]))

        pan_list.append(relocalize_pose[0])
        tilt_list.append(relocalize_pose[1])
        zoom_list.append(relocalize_pose[2])

    save_camera_pose(pan_list, tilt_list, zoom_list, "./", "result.mat")


def baseline_keyframe_based_homography_matching_soccer3():
    sequence = SequenceManager("../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_330",
                               "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_player_bounding_box.mat")

    keyframes = Map('sift')

    keyframes_list = [75, 160, 200, 230, 300]
    pan_list = []
    tilt_list = []
    zoom_list = []

    for i in keyframes_list:
        img = sequence.get_image_gray(index=i, dataset_type=1)
        p, t, z = sequence.ground_truth_pan[i], sequence.ground_truth_tilt[i], sequence.ground_truth_f[i]
        u, v = sequence.camera.principal_point
        c = sequence.camera.camera_center
        r = sequence.camera.base_rotation
        new_keyframe = KeyFrame(img, i, c, r, u, v, p, t, z)
        keyframes.add_keyframe_without_ba(new_keyframe)

    for i in range(0, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=1)
        lost_pose = 60, 0, 3000

        relocalize_pose = relocalization_camera(keyframes, img, lost_pose)

        print("=====The ", i, " iteration=====")

        print("%f" % (relocalize_pose[0] - sequence.ground_truth_pan[i]))
        print("%f" % (relocalize_pose[1] - sequence.ground_truth_tilt[i]))
        print("%f" % (relocalize_pose[2] - sequence.ground_truth_f[i]))

        pan_list.append(relocalize_pose[0])
        tilt_list.append(relocalize_pose[1])
        zoom_list.append(relocalize_pose[2])

    save_camera_pose(pan_list, tilt_list, zoom_list, "./", "result.mat")


if __name__ == "__main__":
    # ut_basketball()
    # ut_soccer3()
    # baseline_keyframe_based_homography_matching_basketball()
    baseline_keyframe_based_homography_matching_soccer3()
    # ut_UBC_hockey()
