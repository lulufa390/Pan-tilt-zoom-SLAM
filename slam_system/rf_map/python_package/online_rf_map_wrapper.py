# online random forest as map
import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
from ctypes import c_char_p
from ctypes import Structure
from ctypes import POINTER
import platform

system = platform.system()

#@todo hardcode library
if system == "Windows":
    lib = cdll.LoadLibrary('C:/graduate_design/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/x64/Debug/rf_map_python.dll')
else:
    lib = cdll.LoadLibrary('/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib')


class OnlineRFMap:
    def __init__(self, rf_file):
        self.rf_file = rf_file

        lib.OnlineRFMap_new.restype = c_void_p
        self.rf_map = lib.OnlineRFMap_new()
        #print('rf_map value 1 {}'.format(self.rf_map))

    def create_map(self, feature_label_file, tree_param_file):
        """
        :param feature_label_file: a .mat file has 'keypoint', 'descriptor' and 'ptz'
        :param tree_param_file:
        :return:
        """

        fl_file = feature_label_file.encode('utf-8')
        tr_file = tree_param_file.encode('utf-8')
        rf_file = self.rf_file.encode('utf-8')
        lib.createOnlineMap.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]

        # print('rf_map point in python {}'.format(self.rf_map))
        #print('rf_map value 2 {}'.format(self.rf_map))
        lib.createOnlineMap(self.rf_map, fl_file, tr_file, rf_file)

    def update_map(self, feature_label_file):
        """
        add a feature label file to the model
        :param feature_label_file:
        :return:
        """

        fl_file = feature_label_file.encode('utf-8')
        rf_file = self.rf_file.encode('utf-8')

        lib.updateOnlineMap.argtypes = [c_void_p, c_char_p, c_char_p]
        lib.updateOnlineMap(self.rf_map, fl_file, rf_file)


    def relocalization(self, feature_location_file, init_pan_tilt_zoom):
        """
        :param feature_file: .mat file has 'keypoint' and 'descriptor'
        :param init_pan_tilt_zoom, 3 x 1, initial camera parameter
        :return:
        """
        feature_location_file = feature_location_file.encode('utf-8')
        test_parameter_file = ''.encode('utf-8')
        pan_tilt_zoom = np.zeros((3, 1))
        for i in range(3):
            pan_tilt_zoom[i] = init_pan_tilt_zoom[i]

        lib.relocalizeCameraOnline.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]

        #print('rf_map value 4 {}'.format(self.rf_map))
        lib.relocalizeCameraOnline(self.rf_map,
                                feature_location_file,
                                test_parameter_file,
                                c_void_p(pan_tilt_zoom.ctypes.data))
        return pan_tilt_zoom

def ut_create_update_map():
    rf_map = OnlineRFMap('debug.txt')

    if system == "Windows":
        pass
        #tree_param_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        #featue_label_files = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/train_feature_file.txt'
    else:
        tree_param_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        feature_label_files = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/train_feature_file.txt'
        f = open(feature_label_files, "r")
        feature_label_files = f.read().splitlines()

    # create the tree by the first .mat file
    rf_map.create_map(feature_label_files[0], tree_param_file)

    # update the tree but .mat files one by one
    for i in range(1, len(feature_label_files)):
        rf_map.update_map(feature_label_files[i])

    #rf_map.create_map(featue_label_files, tree_param_file)


    if system == "Windows":
        feature_location_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/test/bra_mex/17.mat'
    else:
        feature_location_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/test/bra_mex/17.mat'
    init_ptz = np.asarray([11, -9, 3110])
    estimated_ptz = rf_map.relocalization(feature_location_file, init_ptz)
    print('estimated ptz is {}'.format(estimated_ptz))


if __name__ == '__main__':
    ut_create_update_map()

