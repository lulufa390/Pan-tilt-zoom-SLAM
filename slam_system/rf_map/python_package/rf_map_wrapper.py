# random forest as map
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

class RFMap:
    def __init__(self, rf_file):
        self.rf_file = rf_file

        lib.RFMap_new.restype = c_void_p
        self.rf_map = lib.RFMap_new()
        print('rf_map value 1 {}'.format(self.rf_map))

    def create_map(self, feature_label_files, tree_param_file):
        """
        :param tree_param_file:
        :param feature_label_files: .mat file has 'keypoint', 'descriptor' and 'ptz'
        :return:
        """
        fl_file = feature_label_files.encode('utf-8')
        tr_file = tree_param_file.encode('utf-8')
        rf_file = self.rf_file.encode('utf-8')
        lib.createMap.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]

        #print('rf_map point in python {}'.format(self.rf_map))
        print('rf_map value 2 {}'.format(self.rf_map))
        lib.createMap(self.rf_map, fl_file, tr_file, rf_file)
        print('rf_map value 3 {}'.format(self.rf_map))

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

        lib.relocalizeCamera.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]

        print('rf_map value 4 {}'.format(self.rf_map))
        lib.relocalizeCamera(self.rf_map,
                             feature_location_file,
                             test_parameter_file,
                             c_void_p(pan_tilt_zoom.ctypes.data))
        return pan_tilt_zoom


def ut_create_map_relocalization():
    rf_map = RFMap('debug.txt')

    if system == "Windows":
        tree_param_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        featue_label_files = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/train_feature_file.txt'
    else:
        tree_param_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        featue_label_files = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/train_feature_file.txt'

    rf_map.create_map(featue_label_files, tree_param_file)

    if system == "Windows":
        feature_location_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/test/bra_mex/17.mat'
    else:
        feature_location_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/test/bra_mex/17.mat'
    init_ptz = np.asarray([11, -9, 3110])
    estimated_ptz = rf_map.relocalization(feature_location_file, init_ptz)
    print('estimated ptz is {}'.format(estimated_ptz))


if __name__ == '__main__':
    ut_create_map_relocalization()
