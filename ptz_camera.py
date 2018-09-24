import numpy as np

class PTZCamera:
    def __init__(self, principal_point, camera_center, base_rotation):
        self.principal_point = principal_point
        self.camera_center = camera_center
        self.base_rotation = base_rotation

        self.pan = 0.0
        self.tilt = 0.0
        self.focal_lengh = 2000

    def set_ptz(self, ptz):
        # ptz: for pan, tilt and focal length
        pass

    def project_3Dpoint(self, p):
        # p is a 3D point
        pass
    def project_ray(self, ray):
        # ray: pan, tilt in the tripod coordinate (after translation and base rotation)
        pass

    def back_project_to_3D_point(self, x, y):
        # x, y is image pixel location
        # assume z in the world coordinate is 1
        pass

    def back_project_to_ray(self, x, y):
        # x, y is image pixel location
        pass

if __name__ == '__main__':
    # add some unit test in here
    # especially for two back projection function
    pass