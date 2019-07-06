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
            slam.init_system(img, relocalized_camera,bounding_box=bounding_box)

            print("do relocalization!")
        elif slam.new_keyframe:
            slam.add_keyframe(img, slam.current_camera, i, enable_rf=False)
            print("add keyframe!")

        print("=====The ", i, " iteration=====")

        print("%f" % (slam.cameras[i].pan - sequence.ground_truth_pan[i]))
        print("%f" % (slam.cameras[i].tilt - sequence.ground_truth_tilt[i]))
        print("%f" % (slam.cameras[i].focal_length - sequence.ground_truth_f[i]))

        pan_list.append(slam.cameras[i].pan)
        tilt_list.append(slam.cameras[i].tilt)
        zoom_list.append(slam.cameras[i].focal_length)

    for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
        keyframe.save_to_mat(str(i) + ".mat")


    save_camera_pose(pan_list, tilt_list, zoom_list, "./result.mat")


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
    slam.add_keyframe(first_img, first_camera, 0, enable_rf=False)

    pan_list = [first_camera.get_ptz()[0]]
    tilt_list = [first_camera.get_ptz()[1]]
    zoom_list = [first_camera.get_ptz()[2]]

    for i in range(1, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=0)
        bounding_box = sequence.get_bounding_box_mask(i)
        slam.tracking(next_img=img, bad_tracking_percentage=80, bounding_box=bounding_box)

        if slam.tracking_lost:
            relocalized_camera = slam.relocalize(img, slam.current_camera, enable_rf=False)
            slam.init_system(img, relocalized_camera, bounding_box)

            print("do relocalization!")
        elif slam.new_keyframe:
            slam.add_keyframe(img, slam.current_camera, i, enable_rf=False)
            print("add keyframe!")

        print("=====The ", i, " iteration=====")

        print("%f" % (slam.cameras[i].pan - sequence.ground_truth_pan[i]))
        print("%f" % (slam.cameras[i].tilt - sequence.ground_truth_tilt[i]))
        print("%f" % (slam.cameras[i].focal_length - sequence.ground_truth_f[i]))

        pan_list.append(slam.cameras[i].pan)
        tilt_list.append(slam.cameras[i].tilt)
        zoom_list.append(slam.cameras[i].focal_length)

        # slam.keyframe_map.save_keyframes_to_mat("../../map/map_data.mat")

        # for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
        #     keyframe.save_to_mat("../../map/" + str(i) + ".mat")

    # slam.keyframe_map.save_keyframes_to_mat("../../map/map_data.mat")
    save_camera_pose(pan_list, tilt_list, zoom_list, "./result.mat")

    # for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
    #     keyframe.save_to_mat("../../map/" + str(i) + ".mat")


def ut_synthesized():
    sequence = SequenceManager(annotation_path="../../dataset/basketball/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")

    gt_pan, gt_tilt, gt_f = load_camera_pose("../../dataset/synthesized/synthesize_ground_truth.mat", separate=True)

    begin_frame = 3000

    slam = PtzSlam()

    first_img = sequence.get_image_gray(index=begin_frame, dataset_type=2)
    first_camera = sequence.camera
    first_frame_ptz = (gt_pan[begin_frame],
                       gt_tilt[begin_frame],
                       gt_f[begin_frame])
    first_camera.set_ptz(first_frame_ptz)

    pan = [first_frame_ptz[0]]
    tilt = [first_frame_ptz[1]]
    f = [first_frame_ptz[2]]

    slam.init_system(first_img, first_camera)
    # slam.add_keyframe(first_img, first_camera, 0)

    for i in range(3001, 3600):
        img = sequence.get_image_gray(index=i, dataset_type=2)
        # bounding_box = sequence.get_bounding_box_mask(i)
        slam.tracking(next_img=img, bad_tracking_percentage=0)

        # if slam.tracking_lost:
        #     relocalized_camera = slam.relocalize(img, slam.current_camera)
        #     slam.init_system(img, relocalized_camera, bounding_box)
        #
        #     print("do relocalization!")
        # elif slam.new_keyframe:
        #     slam.add_keyframe(img, slam.current_camera, i)
        #     print("add keyframe!")

        print("=====The ", i, " iteration=====")

        print("%f" % (slam.cameras[-1].pan - gt_pan[i]))
        print("%f" % (slam.cameras[-1].tilt - gt_tilt[i]))
        print("%f" % (slam.cameras[-1].focal_length - gt_f[i]))

        pan.append(slam.cameras[-1].pan)
        tilt.append(slam.cameras[-1].tilt)
        f.append(slam.cameras[-1].focal_length)

        # slam.keyframe_map.save_keyframes_to_mat("../../map/map_data.mat")

        # for i, keyframe in enumerate(slam.keyframe_map.keyframe_list):
        #     keyframe.save_to_mat("../../map/" + str(i) + ".mat")

    save_camera_pose(np.array(pan), np.array(tilt), np.array(f),
                     "C:/graduate_design/experiment_result/baseline2/synthesized/ptz-3000.mat")

    # slam.keyframe_map.save_keyframes_to_mat("../../map/map_data.mat")

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

    with open("result2.txt", 'w+') as f:
        for i in range(1, sequence_length):
            next_img = cv.imread("../../UBC_2017/UBC_2017/images/" + filename_list[i],
                                 cv.IMREAD_GRAYSCALE)

            slam.tracking(next_img=next_img, bad_tracking_percentage=40,
                          bounding_box=bounding_box_manager.get_bounding_box_mask(30, 0.5))

            if slam.tracking_lost:
                relocalized_camera = slam.relocalize(next_img, slam.current_camera)
                slam.init_system(next_img, relocalized_camera, bounding_box_manager.get_bounding_box_mask(30, 0.5))
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


def baseline_keyframe_based_homography_matching_synthesized():
    sequence = SequenceManager(annotation_path="../../dataset/basketball/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")

    gt_pan, gt_tilt, gt_f = load_camera_pose("../../dataset/synthesized/synthesize_ground_truth.mat", separate=True)

    keyframes = Map('sift')

    # keyframes_list = [0, 237, 664, 683, 700, 722, 740, 778, 2461]
    keyframes_list = [0, 650, 698, 730, 804]

    pan_list = []
    tilt_list = []
    zoom_list = []

    gt_p_list = []
    gt_tilt_list = []
    gt_f_list = []

    for i in keyframes_list:
        img = sequence.get_image_gray(i, dataset_type=2)
        p, t, z = gt_pan[i], gt_tilt[i], gt_f[i]

        u, v = sequence.camera.principal_point
        c = sequence.camera.camera_center
        r = sequence.camera.base_rotation

        new_keyframe = KeyFrame(img, i, c, r, u, v, p, t, z)

        keyframes.add_keyframe_without_ba(new_keyframe)

    for i in range(0, len(gt_pan), 12):
        img = sequence.get_image_gray(index=i, dataset_type=2)
        lost_pose = 0, 0, 3000

        relocalize_pose = relocalization_camera(keyframes, img, lost_pose)

        print("=====The ", i, " iteration=====")

        print("%f" % (relocalize_pose[0] - gt_pan[i]))
        print("%f" % (relocalize_pose[1] - gt_tilt[i]))
        print("%f" % (relocalize_pose[2] - gt_f[i]))

        pan_list.append(relocalize_pose[0])
        tilt_list.append(relocalize_pose[1])
        zoom_list.append(relocalize_pose[2])

        gt_p_list.append(gt_pan[i])
        gt_tilt_list.append(gt_tilt[i])
        gt_f_list.append(gt_f[i])

    save_camera_pose(pan_list, tilt_list, zoom_list,
    "C:/graduate_design/experiment_result/new/synthesized/homography_keyframe_based/outliers-100/keyframe-40.mat")
    # save_camera_pose(gt_p_list, gt_tilt_list, gt_f_list, "./gt.mat")


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

    for i in range(0, 1200):
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

    save_camera_pose(pan_list, tilt_list, zoom_list, "./result.mat")


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

    save_camera_pose(pan_list, tilt_list, zoom_list, "./result.mat")


def baseline_keyframe_based_homography_matching_UBC_hockey():

    sequence_length = 900
    annotation_mat = sio.loadmat("../../UBC_2017/UBC_2017/UBC_hockey_ground_truth.mat")
    bounding_box_mat_path = "../../UBC_2017/UBC_2017/bounding_box.mat"

    filename_list = ["000" + str(i + 48630) + ".jpg" for i in range(sequence_length)]
    camera_list = annotation_mat["camera"]

    ptz_list = annotation_mat["ptz"]
    image_name_list = annotation_mat["image_name"].squeeze()

    ptz_extend_list = -1 * np.ones([sequence_length, 3])

    bounding_box_manager = SequenceManager(bounding_box_path=bounding_box_mat_path)

    for i, image_name in enumerate(image_name_list, 0):
        index = int(image_name[0][0:-4]) - 48630
        ptz_extend_list[index, 0:3] = ptz_list[i, 0:3]

    camera_center = annotation_mat["cc"].squeeze()
    base_rotation = annotation_mat["base_rotation"].squeeze()

    first_img = cv.imread("../../UBC_2017/UBC_2017/images/" + filename_list[0],
                          cv.IMREAD_GRAYSCALE)

    first_camera = PTZCamera(camera_list[0, 0:2], camera_center, base_rotation)

    first_camera.set_ptz(ptz_extend_list[0])

    keyframes = Map('sift')
    keyframes_list = [30, 150, 330, 510, 820]
    for i in keyframes_list:
        img = cv.imread("../../UBC_2017/UBC_2017/images/" + filename_list[i],
                  cv.IMREAD_GRAYSCALE)

        p, t, z = ptz_extend_list[i][0:3]
        u, v = 640, 360
        c = camera_center

        rotation = np.zeros([3, 3])
        cv.Rodrigues(base_rotation, rotation)
        r = rotation
        new_keyframe = KeyFrame(img, i, c, r, u, v, p, t, z)
        keyframes.add_keyframe_without_ba(new_keyframe)

    with open("result.txt", 'w+') as f:
        for i in range(0, sequence_length):
            next_img = cv.imread("../../UBC_2017/UBC_2017/images/" + filename_list[i],
                                 cv.IMREAD_GRAYSCALE)

            lost_pose = 0, -20, 2400

            relocalize_pose = relocalization_camera(keyframes, next_img, lost_pose)

            print("=====The ", i, " iteration=====")

            print("Pan: %f" % relocalize_pose[0])
            print("Tilt: %f" % relocalize_pose[1])
            print("Zoom: %f" % relocalize_pose[2])

            if not np.all(ptz_extend_list[i] == np.array([-1, -1, -1])):
                print("Pan Error: %f" % (relocalize_pose[0] - ptz_extend_list[i, 0]))
                print("Tilt Error: %f" % (relocalize_pose[1] - ptz_extend_list[i, 1]))
                print("zoom Error: %f" % (relocalize_pose[2] - ptz_extend_list[i, 2]))

                f.write(str(i) + "th frame\n")
                f.write("Pan Error: %f\n" % (relocalize_pose[0] - ptz_extend_list[i, 0]))
                f.write("Tilt Error: %f\n" % (relocalize_pose[1] - ptz_extend_list[i, 1]))
                f.write("zoom Error: %f\n" % (relocalize_pose[2] - ptz_extend_list[i, 2]))


def rf_relocalize_soccer3():
    sequence = SequenceManager("../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_330",
                               "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_player_bounding_box.mat")

    rf_map = RandomForestMap()
    # keyframes = Map('sift')

    # keyframes_list = [75, 160, 200, 230, 300]

    keyframes_list = [i for i in range(0, 330, 10)]

    pan_list = []
    tilt_list = []
    zoom_list = []

    keyframes_obj_list = []

    for i in keyframes_list:
        img = sequence.get_image_gray(index=i, dataset_type=1)
        p, t, z = sequence.ground_truth_pan[i], sequence.ground_truth_tilt[i], sequence.ground_truth_f[i]
        u, v = sequence.camera.principal_point
        c = sequence.camera.camera_center
        r = sequence.camera.base_rotation
        new_keyframe = KeyFrame(img, i, c, r, u, v, p, t, z)

        keyframes_obj_list.append(new_keyframe)
        # rf_map.add_keyframe(new_keyframe)

    rf_map.add_keyframes(keyframes_obj_list)

    for i in range(0, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=1)

        c = sequence.camera.camera_center
        r = sequence.camera.base_rotation
        u = sequence.camera.principal_point[0]
        v = sequence.camera.principal_point[1]
        pan = 60
        tilt = 0
        focal_length = 3000
        relocalize_frame = KeyFrame(img, -1, c, r, u, v, pan, tilt, focal_length)

        ptz = rf_map.relocalize(relocalize_frame)

        print("=====The ", i, " iteration=====")

        print("%f" % (ptz[0] - sequence.ground_truth_pan[i]))
        print("%f" % (ptz[1] - sequence.ground_truth_tilt[i]))
        print("%f" % (ptz[2] - sequence.ground_truth_f[i]))

        pan_list.append(ptz[0])
        tilt_list.append(ptz[1])
        zoom_list.append(ptz[2])

    save_camera_pose(pan_list, tilt_list, zoom_list, "./result.mat")


def rf_relocalize_synthesized():
    sequence = SequenceManager(annotation_path="../../dataset/basketball/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")

    gt_pan, gt_tilt, gt_f = load_camera_pose("../../dataset/synthesized/synthesize_ground_truth.mat", separate=True)

    rf_map = RandomForestMap()
    # keyframes = Map('sift')

    # keyframes_list = [0, 650, 698, 730, 804]
    keyframes_list = [i for i in range(0, 3600, 120)]

    pan_list = []
    tilt_list = []
    zoom_list = []

    gt_p_list = []
    gt_tilt_list = []
    gt_f_list = []

    keyframes_obj_list = []

    has_build_map = False
    if not has_build_map:
        for i in keyframes_list:
            img = sequence.get_image_gray(index=i, dataset_type=2)
            p, t, z = gt_pan[i], gt_tilt[i], gt_f[i]
            u, v = sequence.camera.principal_point
            c = sequence.camera.camera_center
            r = sequence.camera.base_rotation
            new_keyframe = KeyFrame(img, i, c, r, u, v, p, t, z)

            keyframes_obj_list.append(new_keyframe)
            # rf_map.add_keyframe(new_keyframe)

        rf_map.add_keyframes(keyframes_obj_list)

    for i in range(0, len(gt_pan), 12):
        img = sequence.get_image_gray(index=i, dataset_type=2)

        c = sequence.camera.camera_center
        r = sequence.camera.base_rotation
        u = sequence.camera.principal_point[0]
        v = sequence.camera.principal_point[1]
        pan = 0
        tilt = 0
        focal_length = 3000
        relocalize_frame = KeyFrame(img, -1, c, r, u, v, pan, tilt, focal_length)

        ptz = rf_map.relocalize(relocalize_frame)

        print("=====The ", i, " iteration=====")

        print("%f" % (ptz[0] - gt_pan[i]))
        print("%f" % (ptz[1] - gt_tilt[i]))
        print("%f" % (ptz[2] - gt_f[i]))


        pan_list.append(ptz[0])
        tilt_list.append(ptz[1])
        zoom_list.append(ptz[2])

        gt_p_list.append(gt_pan[i])
        gt_tilt_list.append(gt_tilt[i])
        gt_f_list.append(gt_f[i])

    # save_camera_pose(gt_p_list, gt_tilt_list, gt_f_list, "./gt.mat")
    save_camera_pose(pan_list, tilt_list, zoom_list,
    "C:/graduate_design/experiment_result/new/synthesized/homography_keyframe_based/outliers-100/rf-40.mat")



if __name__ == "__main__":
    # ut_basketball()
    # ut_soccer3()
    baseline_keyframe_based_homography_matching_basketball()
    # baseline_keyframe_based_homography_matching_soccer3()
    # ut_UBC_hockey()
    # baseline_keyframe_based_homography_matching_UBC_hockey()
    # rf_relocalize_soccer3()
    # baseline_keyframe_based_homography_matching_synthesized()
    # rf_relocalize_synthesized()
    # ut_synthesized()