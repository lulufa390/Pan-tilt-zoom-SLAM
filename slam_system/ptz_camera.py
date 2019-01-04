"""
PTZCamera class.

Created by Luke, 2018.9
2018.12 add displacement parameter
"""

import numpy as np
import math
import cv2 as cv

import scipy.io as sio


class PTZCamera:
    """
    This is a class for pan-tilt-zoom camera.
    It provides a bunch of functions for projection and reprojection given camera pose.
    """

    def __init__(self, principal_point, camera_center, base_rotation, displacement=None):
        """
        :param principal_point: principal point (u, v).
        :param camera_center: camera projection center.
        :param base_rotation: base rotation matrix [3, 3] array.
        :Param displacement: lambda [6) lambda1, lambda2, ..., lambda6,
        : default [0, 0, 0, 0, 0, 0, 0]
        : displacment between the rotation center and projection center
        """
        if displacement is not None:
            assert len(displacement) == 6

        self.principal_point = principal_point
        self.camera_center = camera_center

        assert base_rotation.shape == (3, 3) or base_rotation.shape == (3,)
        if base_rotation.shape == (3, 3):
            self.base_rotation = base_rotation
        elif base_rotation.shape == (3,):
            self.base_rotation = np.zeros((3, 3))
            cv.Rodrigues(base_rotation, self.base_rotation)

        # set pan, tilt, focal length to default value
        # pan, tilt here are in degree
        self.pan = 0.0
        self.tilt = 0.0
        self.focal_length = 2000

        self.displacement = np.zeros(6)
        if displacement is not None:
            self.displacement = displacement
        self.projection_matrix = np.zeros((3, 4))

    def _compute_camera_matrix(self):
        """
        compute camera matrix
        :return:
        """
        K = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])
        return K

    def _compute_rotation_matrix(self):
        """
        rotation matrix from pan, tilt angles and the base rotation
        :return:
        """
        pan = math.radians(self.pan)
        tilt = math.radians(self.tilt)

        tilt_rot = np.array([[1, 0, 0],
                             [0, math.cos(tilt), math.sin(tilt)],
                             [0, -math.sin(tilt), math.cos(tilt)]])
        pan_rot = np.array([[math.cos(pan), 0, -math.sin(pan)],
                            [0, 1, 0],
                            [math.sin(pan), 0, math.cos(pan)]])
        pan_tilt_rotation = np.dot(tilt_rot, pan_rot)
        rotation = np.dot(pan_tilt_rotation, self.base_rotation)
        return rotation

    def _compute_pan_matrix(self):
        """
        rotation matrix from the pan angle
        :return:
        """

        pan = math.radians(self.pan)
        pan_rot = np.array([[math.cos(pan), 0, -math.sin(pan)],
                            [0, 1, 0],
                            [math.sin(pan), 0, math.cos(pan)]])
        return pan_rot

    def _compute_tilt_matrix(self):
        """
        rotation matrix from the tilt angle
        :return:
        """
        tilt = math.radians(self.tilt)
        tilt_rot = np.array([[1, 0, 0],
                             [0, math.cos(tilt), math.sin(tilt)],
                             [0, -math.sin(tilt), math.cos(tilt)]])
        return tilt_rot

    def _compute_dispalcement(self):
        """
        displacement between the projetion center and the rotation center
        :return:
        """
        fl = self.focal_length
        wt = self.displacement
        return np.array([wt[0] + wt[3] * fl,
                         wt[1] + wt[4] * fl,
                         wt[2] + wt[5] * fl])

    def _recompute_matrix(self):
        """
        compute 3 x 4 projection matrix
        :return:
        """

        K = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])
        rotation = self._compute_rotation_matrix()
        cc = np.identity(4)
        cc[0][3] = -self.camera_center[0]
        cc[1][3] = -self.camera_center[1]
        cc[2][3] = -self.camera_center[2]

        R = np.identity(4)
        R[0:3, 0:3] = rotation

        disp = self._compute_dispalcement()
        disp_mat = np.eye(3, 4)
        disp_mat[0][3] = disp[0]
        disp_mat[1][3] = disp[1]
        disp_mat[2][3] = disp[2]

        self.projection_matrix = np.dot(np.dot(K, disp_mat), np.dot(R, cc))

    def get_ptz(self):
        return np.array([self.pan, self.tilt, self.focal_length])

    def set_ptz(self, ptz):
        """
        Set pan, tilt, focal length for camera.
        :param ptz: array, tuple, or list [3] of pan(in degree), tilt(in degree), focal_length.
        """
        self.pan, self.tilt, self.focal_length = ptz
        self._recompute_matrix()

    def project_3d_point(self, p):
        """
        Project a 3d point in world coordinate to image.
        :param p: 3d point of array [3]
        :return: projected image point tuple(2)
        """
        self._recompute_matrix()
        P = self.projection_matrix
        p_homo = np.array([p[0], p[1], p[2], 1.0])
        uvw = np.dot(P, p_homo)  # 3_4 * 4_1
        assert uvw[2] != 0.0
        return uvw[0] / uvw[2], uvw[1] / uvw[2]

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
                tmp = self.project_3d_point(ps[j])
                if 0 < tmp[0] < width and 0 < tmp[1] < height:
                    points = np.row_stack([points, np.asarray(tmp)])
                    index = np.concatenate([index, [j]], axis=0)
        else:
            for j in range(len(ps)):
                tmp = self.project_3d_point(ps[j])
                points = np.row_stack([points, np.asarray(tmp)])

        return points, index

    def project_ray(self, ray):
        """
        Project a ray in tripod coordinate to image.
        :param ray: ray is a array, tuple or list of [2]
        :return: projected image point tuple(2)
        """
        theta = math.radians(ray[0])
        phi = math.radians(ray[1])

        K = self._compute_camera_matrix()

        pan_tilt_rotation = np.dot(self._compute_tilt_matrix(), self._compute_pan_matrix())
        disp = self._compute_dispalcement()

        ray_p = np.array([math.tan(theta), -math.tan(phi) * math.sqrt(math.tan(theta) * math.tan(theta) + 1), 1])

        img_p = np.dot(K, np.dot(pan_tilt_rotation, ray_p) + disp)
        assert img_p[2] != 0.0

        return img_p[0] / img_p[2], img_p[1] / img_p[2]

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
        Used for general camera (not for ray-based PTZ cameras)
        Back project image point to 3d point.
        The 3d points on the same ray are all corresponding to the image point.
        So you should set a dimension (z) to determine that 3d point.
        :param x: image point location x
        :param y: image point location y
        :return: array [3] of image point
        """

        # set z(3d point) here.
        z = 0

        pan = math.radians(self.pan)
        tilt = math.radians(self.tilt)

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
        pan = math.radians(self.pan)
        tilt = math.radians(self.tilt)

        im_pos = np.array([x, y, 1])  # homogenerous coordinate
        disp = self._compute_dispalcement()

        K = np.array([[self.focal_length, 0, self.principal_point[0]],
                      [0, self.focal_length, self.principal_point[1]],
                      [0, 0, 1]])
        invK = np.linalg.inv(K)

        pan_tilt_R = np.dot(self._compute_tilt_matrix(), self._compute_pan_matrix())
        pan_tilt_R_inv = np.linalg.inv(pan_tilt_R)
        x3d, y3d, z3d = np.dot(pan_tilt_R_inv, np.dot(invK, im_pos) - disp)

        theta = math.atan(x3d / z3d)
        phi = math.atan(-y3d / math.sqrt(x3d * x3d + z3d * z3d))

        return math.degrees(theta), math.degrees(phi)

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


def ut_broadcast_camera_model():
    hockey_model = sio.loadmat("../../ice_hockey_1/ice_hockey_model.mat")
    points = hockey_model['points']
    line_index = hockey_model['line_segment_index']

    annotation = sio.loadmat("../../ice_hockey_1/olympic_2010_reference_frame.mat")
    filename = annotation["images"]
    ptzs = annotation["opt_ptzs"]
    cameras = annotation["opt_cameras"]
    shared_parameters = annotation["shared_parameters"]

    camera = PTZCamera(cameras[0, 0:2], shared_parameters[0:3, 0],
                       shared_parameters[3:6, 0], shared_parameters[6:12, 0])

    for i in range(26):
        img = cv.imread("../../ice_hockey_1/olympic_2010_reference_frame/image/" + filename[i])
        camera.set_ptz(ptzs[i])

        print(camera.projection_matrix)

        image_points = np.ndarray([len(points), 2])

        for j in range(len(points)):
            p = np.array([points[j][0], points[j][1], 0])

            image_points[j][0], image_points[j][1] = camera.project_3d_point(p)


            # ray = np.ndarray(2)
            # p = p - shared_parameters[0:3, 0]
            # p = np.dot(camera.base_rotation, p)
            # ray[0] = math.degrees(math.atan(p[0] / p[2]))
            # ray[1] = math.degrees(math.atan(-p[1] / math.sqrt(p[0] * p[0] + p[2] * p[2])))
            # image_points[j][0], image_points[j][1] = camera.project_ray(ray)

            print(image_points[j])

        # draw lines
        for j in range(len(line_index)):
            begin = line_index[j][0]
            end = line_index[j][1]

            cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                    (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 5)

        cv.imshow("result", img)
        cv.waitKey(0)


def ut_ray_project():
    annotation = sio.loadmat("../../ice_hockey_1/olympic_2010_reference_frame.mat")
    filename = annotation["images"]
    ptzs = annotation["opt_ptzs"]
    cameras = annotation["opt_cameras"]
    shared_parameters = annotation["shared_parameters"]

    camera = PTZCamera(cameras[0, 0:2], shared_parameters[0:3, 0],
                       shared_parameters[3:6, 0], shared_parameters[6:12, 0])
    # camera = PTZCamera(cameras[0, 0:2], shared_parameters[0:3, 0],
    #                    shared_parameters[3:6, 0])

    camera.set_ptz(ptzs[0])
    print(camera.project_ray(np.array([10, 0])))

    print(camera.back_project_to_ray(0, 320))


if __name__ == '__main__':
    ut_broadcast_camera_model()
    # ut_ray_project()