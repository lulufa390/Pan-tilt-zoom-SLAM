"""
PTZCamera class.

Created by Luke, 2018.9
"""

import numpy as np
from math import *


class PTZCamera:
    """
    This is a class for pan-tilt-zoom camera.
    It provides a bunch of functions for projection and reprojection given camera pose.
    """
    def __init__(self, principal_point, camera_center, base_rotation):
        """
        :param principal_point: principal point (u, v).
        :param camera_center: camera projection center.
        :param base_rotation: base rotation matrix [3, 3] array.
        """
        self.principal_point = principal_point
        self.camera_center = camera_center
        self.base_rotation = base_rotation

        # set pan, tilt, focal length to default value
        # pan, tilt here are in degree
        self.pan = 0.0
        self.tilt = 0.0
        self.focal_length = 2000

    def get_ptz(self):
        return np.array([self.pan, self.tilt, self.focal_length])

    def set_ptz(self, ptz):
        """
        Set pan, tilt, focal length for camera.
        :param ptz: array, tuple, or list [3] of pan(in degree), tilt(in degree), focal_length.
        """
        self.pan, self.tilt, self.focal_length = ptz

    def project_3d_point(self, p):
        """
        Project a 3d point in world coordinate to image.
        :param p: 3d point of array [3]
        :return: projected image point tuple(2)
        """
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
        """
        Project a array of 3d points to image.
        :param ps: [n, 3] array of n 3d points in world coordinate.
        :param height: height of image.
        :param width: width of image.
        :return: projected points in image range ([m, 2] array) and its index in ps
        """
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
        """
        Project a ray in tripod coordinate to image.
        :param ray: ray is a array, tuple or list of [2]
        :return: projected image point tuple(2)
        """
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
        """
        Project a array of rays to image.
        :param ps: [n, 2] array of n rays in tripod coordinates.
        :param height: height of image.
        :param width: width of image.
        :return: projected points in image range ([m, 2] array) and its index in ps
        """
        points = np.ndarray([0, 2], np.float32)
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
        """
        Back project image point to 3d point.
        The 3d points on the same ray are all corresponding to the image point.
        So you should set a dimension (z) to determine that 3d point.
        :param x: image point location x
        :param y: image point location y
        :return: array [3] of image point
        """

        # set z(3d point) here.
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

    def back_project_to_3d_points(self, keypoints):
        """
        Back project a bunch of image points to 3d points.
        :param keypoints: [n, 2] array of a array of points on image.
        :return: [n, 3] array of corresponding 3d point.
        """
        points_3d = np.ndarray([0, 3])
        for i in range(len(keypoints)):
            point_3d = self.back_project_to_3d_point(keypoints[i, 0], keypoints[i, 1])
            points_3d = np.row_stack([points_3d, point_3d])
        return points_3d

    def back_project_to_ray(self, x, y):
        """
        Back project image point to ray.
        :param x: image point x.
        :param y: image point y.
        :return: tuple (2) of ray: pan, tilt in degree.
        """
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
        x3d, y3d, z3d = np.dot(inv_mat, np.array([x, y, 1]))

        theta = atan(x3d / z3d)
        phi = atan(-y3d / sqrt(x3d * x3d + z3d * z3d))

        return degrees(theta), degrees(phi)

    def back_project_to_rays(self, points):
        """
        Back project a bunch of image points to rays.
        :param points: [n, 2] array of image points.
        :return: [n, 2] array of corresponding rays.
        """

        rays = np.ndarray([0, 2])
        for i in range(len(points)):
            angles = self.back_project_to_ray(points[i, 0], points[i, 1])
            rays = np.row_stack([rays, angles])
        return rays


if __name__ == '__main__':

    pass
