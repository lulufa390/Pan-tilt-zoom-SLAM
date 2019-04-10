# random forest as map
import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
from ctypes import c_char_p
import platform

system = platform.system()

#@todo hardcode library
if system == "Windows":
    lib = cdll.LoadLibrary('C:/graduate_design/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/x64/Debug/rf_map_python.dll')
else:
    lib = cdll.LoadLibrary('/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib')

class RFMap:
    """
    The map is saved on files.
    """
    def __init__(self, rf_file_name):
        self.rf_file_name = rf_file_name

    def createMap(self, feature_label_files, tree_param_file):
        """
        :param tree_param_file:
        :param feature_label_files: .mat file has 'keypoint', 'descriptor' and 'ptz'
        :return:
        """
        fl_file = feature_label_files.encode('utf-8')
        tr_file = tree_param_file.encode('utf-8')
        rf_file = self.rf_file_name.encode('utf-8')
        lib.createMap.argtypes = [c_char_p, c_char_p, c_char_p]
        lib.createMap(fl_file, tr_file, rf_file)


    def updateMap(self, prev_feature_label_files, feature_label_files):
        pass

    def relocalization(self, feature_location_file):
        """
        :param feature_file: .mat file has 'keypoint' and 'descriptor'
        :return:
        """
        model_name = self.rf_file_name.encode('utf-8')
        feature_location_file = feature_location_file.encode('utf-8')
        test_parameter_file = ''.encode('utf-8')
        pan_tilt_zoom = np.zeros((3, 1))
        lib.relocalizeCamera.argtrypes = [c_char_p, c_char_p, c_char_p, c_void_p]
        #print(model_name)
        #print(feature_location_file)
        #print(test_parameter_file)
        lib.relocalizeCamera(model_name,
                             feature_location_file,
                             test_parameter_file,
                             c_void_p(pan_tilt_zoom.ctypes.data))
        return pan_tilt_zoom


def ut_create_map():
    rf_map = RFMap('debug.txt')

    if system == "Windows":
        tree_param_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        featue_label_files = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/train_feature_file.txt'
    else:
        tree_param_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        featue_label_files = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/train_feature_file.txt'

    for i in range(1):
        rf_map.createMap(featue_label_files, tree_param_file)

def ut_relocalization():
    rf_map = RFMap('debug.txt')
    if system == "Windows":
        feature_location_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/test/bra_mex/17.mat'
    else:
        feature_location_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/test/bra_mex/17.mat'
    estimated_ptz = rf_map.relocalization(feature_location_file)
    print('estimated ptz is {}'.format(estimated_ptz))

    import scipy.io as sio
    data = sio.loadmat(feature_location_file)
    gt_ptz = data['ptz']
    print('ground truth ptz is {}'.format(gt_ptz))

if __name__ == '__main__':
    ut_create_map()
    ut_relocalization()