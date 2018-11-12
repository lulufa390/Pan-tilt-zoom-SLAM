# Ice hockey rink geometry model
import numpy as np
import math
# for plotting
import matplotlib.pyplot as plt

# for load matlab files
import scipy.io as sio
import scipy.misc as misc


class IceHockeyRink:
    def __init__(self):
        self.width = 200  # foot
        self.height = 85
        self.points = []  # two points is a line segment, 0,1 and 2,3

        # for edge point
        self.edge_points = []
        self.edge_point_normals = []

        self.foot_to_meter = 0.3048        
    
    def _add_line(self, x1, y1, x2, y2):
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        self.points.append(a)
        self.points.append(b)


    def _add_edge_point_for_line(self, x1, y1, x2, y2, line_length):
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        d = a - b
        d[0], d[1] = d[1], d[0]
        d[0] *= -1.0  # clockwise, normal direction
        d_norm = np.linalg.norm(d)
        if d_norm != 0:
            d = d/d_norm

        dist = np.linalg.norm(a-b)
        N = int(round(dist/line_length))
        if N == 0 or N == 1:
            mid = (a + b)/2
            self.edge_points.append(mid)
            self.edge_point_normals.append(d)
        else:
            # ignore first and last point
            for i in range(1, N):
                t = i/N
                c = a * t + b *(1-t)
                self.edge_points.append(c)
                self.edge_point_normals.append(d)
    
    def _add_line_with_offset(self, x1, y1, x2, y2, delta_x, delta_y):
        a = np.array([x1+delta_x, y1+delta_y])
        b = np.array([x2+delta_x, y2+delta_y])
        self.points.append(a)
        self.points.append(b)

    def _add_circle(self, x, y, radius):
        c = np.array([x, y])
        for i in range(64):
            theta_1 = 2 * math.pi/64*i
            theta_2 = 2 * math.pi/64*(i+1)
            p1 = np.array([math.cos(theta_1) * radius,math.sin(theta_1) * radius])
            p2 = np.array([math.cos(theta_2) * radius,math.sin(theta_2) * radius])
            p1 += c
            p2 += c
            self._add_line(p1[0], p1[1], p2[0], p2[1])

    def _add_circle_edge_point(self, x, y, radius, edge_length):
        c = np.array([x, y])
        perimeter = 2 * np.pi * radius
        N = int(round(perimeter/edge_length))
        if N < 2:
            N = 2
        for i in range(N):
            theta_1 = 2 * math.pi / N * i
            d = np.array([math.cos(theta_1), math.sin(theta_1)]) # normal direction
            p1 = d * radius + c
            self.edge_points.append(p1)
            self.edge_point_normals.append(d)

    
    def _add_arc(self, x, y, radius, start_angle, stop_angle):
        c = np.array([x, y])
        for i in range(64):
            theta_1 = 2 * math.pi/64*i
            theta_2 = 2 * math.pi/64*(i+1)
            if theta_1 < start_angle or theta_2 > stop_angle:
                continue
            p1 = np.array([math.cos(theta_1) * radius,math.sin(theta_1) * radius])
            p2 = np.array([math.cos(theta_2) * radius,math.sin(theta_2) * radius])
            p1 += c
            p2 += c
            self._add_line(p1[0], p1[1], p2[0], p2[1])
    
    def _add_faceoff(self, x, y):
        w, h = 4, 3
        w_offset, h_offset = 2, 1.5/2

        # horizontal lines
        self._add_line_with_offset(w_offset, h_offset, w_offset+w, h_offset, x, y)
        self._add_line_with_offset(w_offset, -h_offset, w_offset+w, -h_offset, x, y)
        self._add_line_with_offset(-w_offset, h_offset, -(w_offset+w), h_offset, x, y)
        self._add_line_with_offset(-w_offset, -h_offset, -(w_offset+w), -h_offset, x, y)

        # vertical lines
        self._add_line_with_offset(w_offset, h_offset, w_offset, h_offset + h, x, y)
        self._add_line_with_offset(-w_offset, h_offset, -w_offset, h_offset + h, x, y)
        self._add_line_with_offset(w_offset, -h_offset, w_offset, -(h_offset + h), x, y)
        self._add_line_with_offset(-w_offset, -h_offset, -w_offset, -(h_offset + h), x, y)

        
    # generate wire frame of the rink
    def wireframe(self):
        """
        Wire frame for distance based pose optimization
        :return:
        """
        w, h = self.width, self.height       
        half_w, half_h = w/2.0, h/2.0

        # far touch line
        w1, w2, w3, w4 = 11, 64, 25, 28
        h1, h4, h_foot =  11, 28, 0.5
        deleta_1 = 28 - 22.25        
        self._add_line(w4, h, w1+w2, h)
        self._add_line(w1+w2, h, half_w, h)
        self._add_line(half_w, h, half_w+w3, h)
        self._add_line(half_w+w3, h, w - w4, h)

        # center line, left and  blue line        
        self._add_line(half_w-h_foot, 0, half_w-h_foot, h)
        self._add_line(half_w-w3, 0, half_w-w3, h)
        self._add_line(half_w+w3, 0, half_w+w3, h)

        self._add_line(half_w+h_foot, 0, half_w+h_foot, h)
        self._add_line(half_w-w3-1, 0, half_w-w3-1, h)
        self._add_line(half_w+w3+1, 0, half_w+w3+1, h)

        # vertical lines near goal
        self._add_line(w1, deleta_1, w1, h-deleta_1)
        self._add_line(w-w1, deleta_1, w-w1, h-deleta_1)
        self._add_line(0, h4, 0, h-h4)  
        self._add_line(w, h4, w, h-h4)       

        # center and other four circles
        self._add_circle(half_w, half_h, 15)
        self._add_circle(w1 + 20, half_h + 22, 15)
        self._add_circle(w1 + 20, half_h - 22, 15)
        self._add_circle(w - (w1 + 20), half_h + 22, 15)
        self._add_circle(w - (w1 + 20), half_h - 22, 15)

        # four quarter arcs
        delta_theta = math.asin(22.25/28)
        self._add_arc(w4, h4, w4, math.pi, math.pi + delta_theta) #left bottom #math.pi/2*3
        self._add_arc(w4, h-h4, w4, math.pi/2, math.pi) # left top
        self._add_arc(w-w4, h4, w4, math.pi*2 - delta_theta, math.pi*2) # right bottom
        self._add_arc(w-w4, h-h4, w4, 0, math.pi/2)     # right top

        # faceoff lines
        self._add_faceoff(w1+20, half_h-22)
        self._add_faceoff(w1+20, half_h+22)
        self._add_faceoff(w-(w1+20), half_h-22)
        self._add_faceoff(w-(w1+20), half_h+22)

        # goal crease lines
        self._add_line(0, half_h-14, w1, half_h-11)
        self._add_line(0, half_h+14, w1, half_h+11)
        self._add_line(w, half_h-14, w-w1, half_h-11)
        self._add_line(w, half_h+14, w-w1, half_h+11)
       

        # foot to meter        
        self.points = [p*self.foot_to_meter for p in self.points]
        n = len(self.points)
        points = np.zeros((n, 2))
        line_index = np.zeros((n//2, 2))
        for i in range(n):
            points[i][:] = self.points[i]
        for i in range(n//2):
            line_index[i][0] = i * 2
            line_index[i][1] = i * 2 + 1
        return (points, line_index)

    def template2D(self):
        """
        visualize in GUI program
        :return:
        """
        w, h = self.width, self.height
        half_w, half_h = w / 2.0, h / 2.0
        deleta_1 = 28 - 22.25

        # far touch line
        w1, w2, w3, w4 = 11, 64, 25, 28
        h1, h4, h_foot = 11, 28, 0.5
        self._add_line(w4, h, w1 + w2, h)
        self._add_line(w1 + w2, h, half_w, h)
        self._add_line(half_w, h, half_w + w3, h)
        self._add_line(half_w + w3, h, w - w4, h)

        # near touch line
        self._add_line(w4, 0, w1 + w2, 0)
        self._add_line(w1 + w2, 0, half_w, 0)
        self._add_line(half_w, 0, half_w + w3, 0)
        self._add_line(half_w + w3, 0, w - w4, 0)

        # center line, left and  blue line
        self._add_line(half_w - h_foot, 0, half_w - h_foot, h)
        self._add_line(half_w - w3, 0, half_w - w3, h)
        self._add_line(half_w + w3, 0, half_w + w3, h)

        self._add_line(half_w + h_foot, 0, half_w + h_foot, h)
        self._add_line(half_w - w3 - 1, 0, half_w - w3 - 1, h)
        self._add_line(half_w + w3 + 1, 0, half_w + w3 + 1, h)


        # vertical lines near goal
        self._add_line(w1, deleta_1, w1, h - deleta_1)
        self._add_line(w - w1, deleta_1, w - w1, h - deleta_1)
        self._add_line(0, h4, 0, h - h4)
        self._add_line(w, h4, w, h - h4)

        # center and other four circles
        self._add_circle(half_w, half_h, 15)
        self._add_circle(w1 + 20, half_h + 22, 15)
        self._add_circle(w1 + 20, half_h - 22, 15)
        self._add_circle(w - (w1 + 20), half_h + 22, 15)
        self._add_circle(w - (w1 + 20), half_h - 22, 15)

        # four quarter arcs
        delta_theta = math.asin(22.25 / 28)  # not used
        self._add_arc(w4, h4, w4, math.pi, math.pi + math.pi / 2)  # left bottom #math.pi/2*3
        self._add_arc(w4, h - h4, w4, math.pi / 2, math.pi)  # left top
        self._add_arc(w - w4, h4, w4, math.pi * 2 - math.pi / 2, math.pi * 2)  # right bottom
        self._add_arc(w - w4, h - h4, w4, 0, math.pi / 2)  # right top

        # faceoff lines
        self._add_faceoff(w1 + 20, half_h - 22)
        self._add_faceoff(w1 + 20, half_h + 22)
        self._add_faceoff(w - (w1 + 20), half_h - 22)
        self._add_faceoff(w - (w1 + 20), half_h + 22)

        # goal crease lines
        self._add_line(0, half_h - 14, w1, half_h - 11)
        self._add_line(0, half_h + 14, w1, half_h + 11)
        self._add_line(w, half_h - 14, w - w1, half_h - 11)
        self._add_line(w, half_h + 14, w - w1, half_h + 11)

        # goal box line, 8 x 4 box
        self._add_line(w1, half_h - 4, w1 + 2, half_h - 4)
        self._add_line(w1, half_h + 4, w1 + 2, half_h + 4)
        self._add_line(w - w1, half_h - 4, w - (w1 + 2), half_h - 4)
        self._add_line(w - w1, half_h + 4, w - (w1 + 2), half_h + 4)

        # four circle in neutral zone
        w5, h5 = 20, 22
        self._add_arc(half_w - w5, half_h + h5, 2, 0, math.pi * 2)
        self._add_arc(half_w - w5, half_h - h5, 2, 0, math.pi * 2)
        self._add_arc(half_w + w5, half_h + h5, 2, 0, math.pi * 2)
        self._add_arc(half_w + w5, half_h - h5, 2, 0, math.pi * 2)

        # foot to meter
        self.points = [p * self.foot_to_meter for p in self.points]
        n = len(self.points)
        points = np.zeros((n, 2))
        line_index = np.zeros((n // 2, 2))
        for i in range(n):
            points[i][:] = self.points[i]
        for i in range(n // 2):
            line_index[i][0] = i * 2
            line_index[i][1] = i * 2 + 1
        return (points, line_index)

    def edgePoints(self):
        """
        sampled edge point for model tracking
        :return:
        """
        w, h = self.width, self.height
        half_w, half_h = w / 2.0, h / 2.0
        edge_length = 1.0 # feet


        w1, w2, w3, w4 = 11, 64, 25, 28
        h1, h4, h_foot = 11, 28, 0.5
        deleta_1 = 28 - 22.25

        # far touch line
        self._add_edge_point_for_line(w4, h, w1 + w2, h, edge_length)
        self._add_edge_point_for_line(w1 + w2, h, half_w, h, edge_length)
        self._add_edge_point_for_line(half_w, h, half_w + w3, h, edge_length)
        self._add_edge_point_for_line(half_w + w3, h, w - w4, h, edge_length)


        # near touch line
        #self._add_edge_point_for_line(w4, 0, w1 + w2, 0, edge_length)
        #self._add_edge_point_for_line(w1 + w2, 0, half_w, 0, edge_length)
        #self._add_edge_point_for_line(half_w, 0, half_w + w3, 0, edge_length)
        #self._add_edge_point_for_line(half_w + w3, 0, w - w4, 0, edge_length)

        # center line, left and  blue line
        self._add_edge_point_for_line(half_w, 0, half_w, h, edge_length)
        self._add_edge_point_for_line(half_w - w3, 0, half_w - w3, h, edge_length)
        self._add_edge_point_for_line(half_w + w3, 0, half_w + w3, h, edge_length)

        # vertical lines near goal
        self._add_edge_point_for_line(w1, deleta_1, w1, h - deleta_1, edge_length)
        self._add_edge_point_for_line(w - w1, deleta_1, w - w1, h - deleta_1, edge_length)
        self._add_edge_point_for_line(0, h4, 0, h - h4, edge_length)
        self._add_edge_point_for_line(w, h4, w, h - h4, edge_length)

        # center and other four circles
        self._add_circle_edge_point(half_w, half_h, 15, edge_length)
        self._add_circle_edge_point(w1 + 20, half_h + 22, 15, edge_length)
        self._add_circle_edge_point(w1 + 20, half_h - 22, 15, edge_length)
        self._add_circle_edge_point(w - (w1 + 20), half_h + 22, 15, edge_length)
        self._add_circle_edge_point(w - (w1 + 20), half_h - 22, 15, edge_length)

        # foot to meter
        self.edge_points = [p * self.foot_to_meter for p in self.edge_points]
        n = len(self.edge_points)
        edge_points = np.zeros((n, 2))
        edge_point_normals = np.zeros((n, 2))
        for i in range(n):
            edge_points[i][:] = self.edge_points[i]
            edge_point_normals[i][:] = self.edge_point_normals[i]
        return (edge_points, edge_point_normals)

        
    # generate grid point on the field surface
    def gridpoint(self, unit):
        w, h = self.width, self.height
        w_num, h_num = round(w/unit), round(h/unit)
        n = w_num * h_num
        points = np.zeros((n, 2))
        for h in range(h_num):
            for w in range(w_num):
                index, x, y = h * w_num + w, unit*w, unit*h
                points[index][:] = x, y
        points = [p*self.foot_to_meter for p in points]
        return points

def ut_template2D():
    model = IceHockeyRink()
    points, line_segment_index = model.template2D()

    N = line_segment_index.shape[0]
    print(N)

    idx1 = line_segment_index[:, 0].astype('int')
    idx2 = line_segment_index[:, 1].astype('int')
    x = [points[idx1, 0], points[idx2, 0]]
    y = [points[idx1, 1], points[idx2, 1]]

    fig = plt.figure()
    plt.plot(x, y, color='k')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    # save points and line segment index to a .txt file
    f = open('ice_hockey_model.txt', 'w')
    n1, n2 = points.shape[0], line_segment_index.shape[0]
    f.write('%d %d\n' % (n1, n2))
    for i in range(n1):
        f.write('%lf %lf\n' % (points[i][0], points[i][1]))
    for i in range(n2):
        f.write('%d %d\n' % (line_segment_index[i][0], line_segment_index[i][1]))
    f.close()

def draw_template():
    model = IceHockeyRink()
    points, line_segment_index = model.template2D()
    N = line_segment_index.shape[0]
    print(N)

    def world2Image(x, y):
        x *= 12
        y *= 12
        y = 85 * 12 - y

        x += 70
        y += 70
        return (x, y)

    for i in range(points.shape[0]):
        x, y = points[i,:]
        x, y = world2Image(x, y)
        points[i, :] = x, y

    import cv2 as cv
    h, w = 1160, 2540
    blank_image = np.ones((h, w, 3), np.uint8) * 255
    for i in range(N):
        idx1, idx2 = line_segment_index[i,:].astype('int')
        x1, y1 = points[idx1,:].astype('int')
        x2, y2 = points[idx2,:].astype('int')
        cv.line(blank_image, (x1, y1), (x2, y2), (0, 0, 0), 5)

    #cv.imshow("ice hockey rink", blank_image)
    cv.imwrite('ice_hockey_template.png', blank_image)



def ut_wireframe():
    model = IceHockeyRink()
    points, line_index = model.wireframe()
    grid_points = model.gridpoint(4)
    sio.savemat('ice_hockey_model.mat', {'points': points, 'line_segment_index': line_index,
                                         'grid_points': grid_points})

def ut_edge_point():
    model = IceHockeyRink()
    edge_points, edge_point_normal_directions = model.edgePoints()

    sio.savemat('ice_hockey_edge_point.mat',
                {'edge_points': edge_points,
                 'edge_point_normal_directions': edge_point_normal_directions})

    N = edge_points.shape[0]
    print(N)


    # visualize and debug, edge point
    x, y = edge_points[:, 0], edge_points[:, 1]
    fig = plt.figure()
    plt.plot(x, y, 'k.')
    # edge normal point
    edge_normal_points = np.zeros((N*2, 2))
    for i in range(N):
        p1 = edge_points[i] + edge_point_normal_directions[i] * 0.5
        p2 = edge_points[i] - edge_point_normal_directions[i] * 0.5
        edge_normal_points[2*i] = p1
        edge_normal_points[2*i+1] = p2
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')

    plt.axis('equal')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    ut_edge_point()
    #ut_template2D()
    #draw_template()





       



        
    
    












    
