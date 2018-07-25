from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random


fig = plt.figure(num=1, figsize=(10,5))
ax = fig.add_subplot(111, projection='3d',)


x = np.linspace(0, 10, 10)
y = np.linspace(0, 10, 10)
z = np.linspace(0, 10, 10)


soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")

line_index = soccer_model['line_segment_index']
points = soccer_model['points']




# print(line_index)
# print(points)

# Make data
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))



for i in range(len(line_index)):
    x = [points[line_index[i][0]][0], points[line_index[i][1]][0]]
    y = [points[line_index[i][0]][1], points[line_index[i][1]][1]]
    z = [0,0]

    # print(x, y, z)
    ax.plot(x, y, z, color = 'g')
    # break

# Plot the surface

ax.set_xlim(0, 120)
ax.set_ylim(0, 70)
ax.set_zlim(0, 10)

px = []
py = []
pz = []


for i in range(200):
    xside = random.randint(0,1)
    px.append( xside * random.gauss(0, 5) + (1-xside) * random.gauss(108,5) )
    py.append( random.uniform(0,70) )
    pz.append( random.uniform(0,10) )

    px.append( random.uniform(0,108) )
    py.append( random.gauss(63, 2) )
    pz.append( random.uniform(0,10) )

for i in range(0,50):
    tmpx = random.gauss(54, 20)
    while tmpx > 108 or tmpx < 0:
        tmpx = random.gauss(54, 20)
    # px.append(random.gauss(54, 25))
    tmpy = random.gauss(32,20)
    while tmpy > 63 or tmpy < 0:
        tmpy = random.gauss(32, 20)

    px.append(tmpx)
    py.append(tmpy)

    # py.append(random.gauss(32, 25))
    pz.append(random.uniform(0, 1))


ax.scatter(px, py, pz, color='r', marker='o')

# plt.xticks(np.linspace(0, 120, 50))

# ax.set_xscale(2)
# ax.plot(x, y, -z)
# ax.plot_surface(x, y, z, color='b')

plt.show()


