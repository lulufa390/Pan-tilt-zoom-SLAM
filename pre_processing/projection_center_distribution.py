"""
Show the figure 6 in Mimicking Human Camera Operators.

Created by Luke, 2019.1.6
"""

import scipy.io as sio
import numpy as np

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt


def distribution_image(path):
    annotation = sio.loadmat(path)
    ptzs = annotation["opt_ptzs"]
    shared_parameters = annotation["shared_parameters"]

    # Get the sequence length
    N = ptzs.shape[0]

    rotation_center = shared_parameters[0:3, 0]
    displace_paras = shared_parameters[6:12, 0]

    avg_project_center = np.zeros(3)

    project_centers = []

    for i in range(N):
        fl = ptzs[i, 2]
        displacement = np.array([displace_paras[0] + displace_paras[3] * fl,
                                 displace_paras[1] + displace_paras[4] * fl,
                                 displace_paras[2] + displace_paras[5] * fl])

        project_center = rotation_center + displacement
        project_centers.append(project_center)
        avg_project_center += project_center

    avg_project_center /= N

    points = np.ndarray([N, 3])

    for i, center in enumerate(project_centers):
        points[i, 0:3] = center - avg_project_center

    # plot for Y-X
    plt.figure("project center distribution Y-X")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.scatter(points[:, 0], points[:, 1])

    # plot for Z-Y
    plt.figure("project center distribution Z-Y")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.legend(loc="best")
    plt.scatter(points[:, 1], points[:, 2])

    plt.show()


if __name__ == "__main__":
    distribution_image("../../ice_hockey_1/olympic_2010_reference_frame.mat")
