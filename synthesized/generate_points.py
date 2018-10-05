from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
from math import *
from transformation import TransFunction


class DataSynthesize:
    def __init__(self):
        """
        load the soccer field model
        """
        soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")
        self.line_index = soccer_model['line_segment_index']
        self.points = soccer_model['points']

        """         
        load the sequence annotation     
        """
        seq = sio.loadmat("./two_point_calib_dataset/highlights/seq3_anno.mat")
        self.annotation = seq["annotation"]
        self.meta = seq['meta']

        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.proj_center = self.meta[0][0]["cc"][0]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)

    @staticmethod
    def generate_points(num):
        list_pts = []

        # fix the random seed
        random.seed(1)
        for i in range(num):
            choice = random.randint(0, 6)
            if choice < 3:
                x_side = random.randint(0, 1)
                list_pts.append([x_side * random.gauss(0, 5) + (1 - x_side) * random.gauss(108, 5),
                                 random.uniform(0, 70), random.uniform(0, 10)])
            elif choice < 5:
                list_pts.append([random.uniform(0, 108), random.gauss(63, 2), random.uniform(0, 10)])
            else:
                tmp_x = random.gauss(54, 20)
                while tmp_x > 108 or tmp_x < 0:
                    tmp_x = random.gauss(54, 20)

                tmp_y = random.gauss(32, 20)
                while tmp_y > 63 or tmp_y < 0:
                    tmp_y = random.gauss(32, 20)

                list_pts.append([tmp_x, tmp_y, random.uniform(0, 1)])

        pts_arr = np.array(list_pts, dtype=np.float32)
        return pts_arr

    def show_image_sequence(self, pts):

        for i in range(self.annotation.size):
            img = np.zeros((720, 1280, 3), np.uint8)
            img.fill(255)

            pan, tilt, f = self.annotation[0][i]['ptz'].squeeze()

            self.draw_soccer_line(img, pan, tilt, f)

            # draw the feature points in images
            for j in range(len(pts)):
                p = np.array(pts[j])

                res = TransFunction.from_3d_to_2d(self.u, self.v, f, pan, tilt, self.proj_center, self.base_rotation, p)
                res2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, rays[j][0], rays[j][1])

                if 0 < res[0] < 1280 and 0 < res[1] < 720:
                    print(p)
                    print("ray", rays[j][0], rays[j][1])
                    print("res:, ", res)
                    print("res2: ", res2)
                    print("==========")

                cv.circle(img, (int(res[0]), int(res[1])), color=(0, 0, 0), radius=8, thickness=2)
                cv.circle(img, (int(res2[0]), int(res2[1])), color=(255, 0, 0), radius=8, thickness=2)

            cv.imshow("synthesized image", img)
            cv.waitKey(0)

    @staticmethod
    def save_to_mat(pts, rays):
        key_points = dict()
        features = []

        # generate features randomly
        for i in range(len(pts)):
            vec = np.random.random(16)
            vec = vec.reshape(1, 16)
            vec = normalize(vec, norm='l2')
            vec = np.squeeze(vec)
            features.append(vec)

        key_points['features'] = features
        key_points['pts'] = pts
        key_points['rays'] = rays

        sio.savemat('synthesize_data.mat', mdict=key_points)

    """
    this function is to draw the 3D model of soccer field
    """

    def draw_3d_model(self, features):
        plt.ion()
        fig = plt.figure(num=1, figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 70)
        ax.set_zlim(0, 10)
        for i in range(len(self.line_index)):
            x = [self.points[self.line_index[i][0]][0], self.points[self.line_index[i][1]][0]]
            y = [self.points[self.line_index[i][0]][1], self.points[self.line_index[i][1]][1]]
            z = [0, 0]
            ax.plot(x, y, z, color='g')

        ax.scatter(features[:, 0], features[:, 1], features[:, 2], color='r', marker='o')
        plt.show()

    """
    this function draws the lines of soccer field in one image from a specific camera pose
    """

    def draw_soccer_line(self, img, p, t, f):
        k = np.array([[f, 0, self.u], [0, f, self.v], [0, 0, 1]])

        pan = radians(p)
        tilt = radians(t)

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),
                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))
        rotation = np.dot(rotation, self.base_rotation)

        image_points = np.ndarray([len(self.points), 2])

        for j in range(len(self.points)):
            p = np.array([self.points[j][0], self.points[j][1], 0])
            p = np.dot(k, np.dot(rotation, p - self.proj_center))
            image_points[j][0] = p[0] / p[2]
            image_points[j][1] = p[1] / p[2]

        for j in range(len(self.line_index)):
            begin = self.line_index[j][0]
            end = self.line_index[j][1]
            cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                    (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 5)

    def computer_all_ray(self, points):
        all_rays = []
        for i in range(0, len(points)):
            ray = TransFunction.compute_rays(self.proj_center, pts[i], self.base_rotation)
            all_rays.append(ray)
        return all_rays


synthesize = DataSynthesize()
pts = synthesize.generate_points(300)
synthesize.draw_3d_model(pts)
rays = synthesize.computer_all_ray(pts)
synthesize.show_image_sequence(pts)
synthesize.save_to_mat(pts, rays)
