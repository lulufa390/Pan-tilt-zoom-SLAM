import numpy as np
from key_frame import KeyFrame
from util import overlap_pan_angle
from bundle_adjustment import bundle_adjustment


from sequence_manager import SequenceManager

class Map:
    def __init__(self):
        # [N, 2] float64 array
        self.global_ray = []

        # list of KeyFrame object
        self.keyframe_list = []

    def add_first_keyframe(self, keyframe, verbose = False):
        """
        add first key frame, no bundle adjustment
        :param keyframe:
        :return:
        """
        assert isinstance(keyframe, KeyFrame)
        self.keyframe_list = []
        self.keyframe_list.append(keyframe)
        if verbose:
            print('first key frame is added, no bundle adjustment and landmark')

    def add_keyframe_without_ba(self, keyframe, verbose = False):
        """
        add keyframe without bundle adjustment
        :param keyframe:
        :param verbose:
        :return:
        """
        assert isinstance(keyframe, KeyFrame)
        self.keyframe_list.append(keyframe)

    def add_keyframe_with_ba(self, keyframe, save_path, verbose = False):
        """
        add one keyframe and do bundle adjustment
        It will take a long time
        :param keyframe:
        :param: save_path
        :param verbose:
        :return: updated map, please note the original map is updated
        """
        # check parameter
        assert isinstance(keyframe, KeyFrame)
        assert len(self.keyframe_list) >= 1

        # step 1: get common camera parameters from the first frame
        ref_frame = self.keyframe_list[0]

        camera_center = ref_frame.center
        base_rotation = ref_frame.base_rotation
        u, v = ref_frame.u, ref_frame.v

        # step 2: prepare data for bundle adjustment
        self.add_keyframe_without_ba(keyframe, False)

        N = len(self.keyframe_list)
        images = []
        image_indices = []
        initial_ptzs = np.zeros((N, 3))

        for i in range(N):
            keyframe = self.keyframe_list[i]
            images.append(keyframe.img)
            image_indices.append(keyframe.img_index)
            initial_ptzs[i] = keyframe.pan, keyframe.tilt, keyframe.f

        # step 3: bundle adjustment
        landmarks, keyframes = bundle_adjustment(images, image_indices, initial_ptzs, camera_center, base_rotation, u, v, save_path, verbose)

        # remove the last keyframe as it is not used in bundle adjustment
        self.keyframe_list.pop()

        # check if keyframe has landmarks
        self.global_ray = landmarks
        self.keyframe_list = []
        for i in range(len(keyframes)):
            keyframe = keyframes[i]
            if keyframe.get_feature_num() > 0:
                self.keyframe_list.append(keyframe)
            else:
                print('warning: key frame, %d, image index %d is not included in the map' % (i, image_indices[i]))

        if verbose:
            print('updated map, number of key frame: %d, number of landmark %d', len(self.keyframe_list), len(landmarks))
        return landmarks, self.keyframe_list


    def good_new_keyframe(self, ptz, threshold1 = 5, threshold2 = 20, im_width = 1280, verbose = False):
        """
        good or not as a new keyframe, small overlap with all existing keyframes
        :param ptz: array [3] camera pose
        :threshold1, pan angle overlap in degree, lower bound, too small, no shared features
        :threshold2, pan angle overlap in degree, upp bound, an image covers about 30 degrees
        :return: True for good new key frame, false for not
        """
        # check parameter
        assert ptz.shape[0] == 3
        N = len(self.keyframe_list)
        if N == 0:
            print('Warning: not existing key frames')
            return False

        map_ptzs = np.zeros((N, 3))
        for i in range(N):
            map_ptzs[i][0] = self.keyframe_list[i].pan
            map_ptzs[i][1] = self.keyframe_list[i].tilt
            map_ptzs[i][2] = self.keyframe_list[i].f

        pan_angle_overlaps = np.zeros(N)
        for i in range(N):
            pan_angle_overlaps[i] = overlap_pan_angle(ptz[2], ptz[0], map_ptzs[i][2], map_ptzs[i][0], im_width)

        if verbose:
            print('candidate key frame overlap: ', pan_angle_overlaps)

        max_overlap = max(pan_angle_overlaps)
        print("overlap", max_overlap)
        return max_overlap > threshold1 and max_overlap < threshold2

def ut_add_first_key_frame():
    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    camera_center = input.get_camera_center()
    base_rotation = input.get_base_rotation()
    u = 1280 / 2
    v = 720 / 2

    image_index = [0]  # 680, 690, 700, 730, 800
    im = input.get_basketball_image(image_index[0])
    ptz = input.get_ptz(image_index[0])
    keyframe = KeyFrame(im, image_index[0], camera_center, base_rotation, u, v, ptz[0], ptz[1], ptz[2])

    a_map = Map()
    a_map.add_first_keyframe(keyframe, True)

def ut_good_new_keyframe():
    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    camera_center = input.get_camera_center()
    base_rotation = input.get_base_rotation()
    u = 1280 / 2
    v = 720 / 2

    image_index = [0]  # 680, 690, 700, 730, 800
    im = input.get_basketball_image(image_index[0])
    ptz = input.get_ptz(image_index[0])
    keyframe = KeyFrame(im, image_index[0], camera_center, base_rotation, u, v, ptz[0], ptz[1], ptz[2])

    a_map = Map()
    a_map.add_first_keyframe(keyframe, False)

    # test the result frames
    for i in range(0, 3600, 1):
        ptz = input.get_ptz(i)
        keyframe = KeyFrame(im, i, camera_center, base_rotation, u, v, ptz[0], ptz[1], ptz[2])

        if a_map.good_new_keyframe(ptz, 3, 20, 1280, False):
            a_map.add_keyframe_without_ba(keyframe)
            print('add key frame from index %d, pan angle %f' % (i, ptz[0]))

    print('number of keyframe is %d' % (len(a_map.keyframe_list)))

def ut_add_keyframe_with_ba():
    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    camera_center = input.get_camera_center()
    base_rotation = input.get_base_rotation()
    u = 1280 / 2
    v = 720 / 2

    image_index = [0]  # 680, 690, 700, 730, 800
    im = input.get_basketball_image(image_index[0])
    ptz = input.get_ptz(image_index[0])
    keyframe = KeyFrame(im, image_index[0], camera_center, base_rotation, u, v, ptz[0], ptz[1], ptz[2])

    a_map = Map()
    a_map.add_first_keyframe(keyframe, False)

    # test the result frames
    for i in range(1, 3600, 5):
        ptz = input.get_ptz(i)
        im = input.get_basketball_image(i)
        keyframe = KeyFrame(im, i, camera_center, base_rotation, u, v, ptz[0], ptz[1], ptz[2])

        if a_map.good_new_keyframe(ptz, 10, 25, 1280, False):
            print('add key frame from index %d, pan angle %f' % (i, ptz[0]))
            a_map.add_keyframe_with_ba(keyframe, '.', True)


    print('number of keyframe is %d' % (len(a_map.keyframe_list)))







if __name__ == '__main__':
    #ut_add_first_key_frame()
    #ut_good_new_keyframe()
    ut_add_keyframe_with_ba()





