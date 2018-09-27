import numpy as np
from math import *


class PTZCamera:
    def __init__(self, principal_point, camera_center, base_rotation):
        self.principal_point = principal_point
        self.camera_center = camera_center
        self.base_rotation = base_rotation

        self.pan = 0.0
        self.tilt = 0.0
        self.focal_length = 2000

    def set_ptz(self, ptz):
        self.pan, self.tilt, self.focal_length = ptz

    def project_3d_point(self, p):
        # p is a 3D point
        pan = radians(self.pan)
        tilt = radians(self.tilt)

        k = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        rotation = np.dot(rotation, self.base_rotation)

        position = np.dot(k, np.dot(rotation, p - self.camera_center))

        return position[0] / position[2], position[1] / position[2]

    def project_3d_points(self, ps, height=0, width=0):
        points = np.ndarray([0, 2])
        index = np.ndarray([0])

        if height != 0 and width != 0:
            for j in range(len(ps)):
                tmp = self.project_3Dpoint(ps[j])
                if 0 < tmp[0] < width and 0 < tmp[1] < height:
                    points = np.row_stack([points, np.asarray(tmp)])
                    index = np.concatenate([index, [j]], axis=0)
        else:
            for j in range(len(ps)):
                tmp = self.project_3Dpoint(ps[j])
                points = np.row_stack([points, np.asarray(tmp)])

        return points, index

    def project_ray(self, ray):
        theta = radians(ray[0])
        phi = radians(ray[1])
        pan = radians(self.pan)
        tilt = radians(self.tilt)

        k = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        p = np.array([tan(theta), -tan(phi) * sqrt(tan(theta) * tan(theta) + 1), 1])

        position = np.dot(k, np.dot(rotation, p))

        return position[0] / position[2], position[1] / position[2]

    def project_rays(self, rays, height=0, width=0):
        points = np.ndarray([0, 2])
        index = np.ndarray([0])

        if height != 0 and width != 0:
            for j in range(len(rays)):
                tmp = self.project_ray(rays[j])
                if 0 < tmp[0] < width and 0 < tmp[1] < height:
                    points = np.row_stack([points, np.asarray(tmp)])
                    index = np.concatenate([index, [j]], axis=0)
        else:
            for j in range(len(rays)):
                tmp = self.project_ray(rays[j])
                points = np.row_stack([points, np.asarray(tmp)])

        return points, index

    def back_project_to_3d_point(self, x, y):
        z = 0
        pan = radians(self.pan)
        tilt = radians(self.tilt)

        k = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        rotation = np.dot(rotation, self.base_rotation)

        inv_mat = np.linalg.inv(np.dot(k, rotation))

        coe = (z - self.camera_center[2]) / (inv_mat[2, 0] * x + inv_mat[2, 1] * y + inv_mat[2, 2])

        p = np.dot(inv_mat, coe * np.array([x, y, 1])) + self.camera_center

        return p

    def back_project_to_ray(self, x, y):
        pan = radians(self.pan)
        tilt = radians(self.tilt)

        k = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        inv_mat = np.linalg.inv(np.dot(k, rotation))

        # rotation = np.linalg.inv(rotation)

        x3d, y3d, z3d = np.dot(inv_mat, np.array([x, y, 1]))

        theta = atan(x3d / z3d)
        phi = atan(-y3d / sqrt(x3d * x3d + z3d * z3d))

        return degrees(theta), degrees(phi)

    def back_project_to_rays(self, points):
        rays = np.ndarray([0, 2])
        for i in range(len(points)):
            angles = self.back_project_to_ray(points[i, 0], points[i, 1])
            rays = np.row_stack([rays, angles])
        return rays


if __name__ == '__main__':
    # add some unit test in here
    # especially for two back projection function
    pass
