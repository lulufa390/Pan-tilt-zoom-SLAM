# Ice hockey rink geometry model
import numpy as np
import math
# for plotting
import matplotlib.pyplot as plt

# for load matlab files
import scipy.io as sio
import scipy.misc as misc


class BasketballCourt:
    def __init__(self):
        self.width = 94  # foot
        self.height = 50
        self.points = []  # two points is a line segment, 0,1 and 2,3
        self.foot_to_meter = 0.3048        
    
    def _add_line(self, x1, y1, x2, y2):
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        self.points.append(a)
        self.points.append(b)
    
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

        
    # generate wire frame of the rink
    def wireframe(self):        
        w, h = self.width, self.height       
        half_w, half_h = w/2.0, h/2.0


        # two side lines
        self._add_line(0, 0, half_w, 0)
        self._add_line(half_w, 0, w, 0)
        self._add_line(0, h, half_w, h)
        self._add_line(half_w, h, w, h)

        # two touch lines (end lines) and one middle line
        self._add_line(0, 0, 0, h)
        self._add_line(half_w, 0, half_w, h)
        self._add_line(w, 0, w, h)

        # box under basket, left, right
        h1, w1 = (50 - 12) / 2, 19
        self._add_line(0, h1, w1, h1)
        self._add_line(0, h-h1, w1, h-h1)
        self._add_line(w1, h1, w1, h-h1)

        self._add_line(w, h1, w-w1, h1)
        self._add_line(w, h - h1, w-w1, h - h1)
        self._add_line(w-w1, h1, w-w1, h - h1)

        # two half circle, under basket
        self._add_arc(w1, half_h, 6, 0, math.pi/2)
        self._add_arc(w1, half_h, 6,  math.pi / 2 * 3, 2*math.pi)
        self._add_arc(w-w1, half_h, 6, math.pi / 2, math.pi / 2 * 3)

        # foot to meter        
        self.points = [p*self.foot_to_meter for p in self.points]
        n = len(self.points)
        points = np.zeros((n, 2))
        line_index = np.zeros((n//2, 2))
        for i in range(n):
            points[i][:] = self.points[i]
        for i in range(n//2):
            line_index[i][0] = i * 2
            line_index[i][1] = i * 2 +1
        return (points, line_index)
        
    # generate grid point on the field surface
    def gridpoint(self, unit):
        w, h = self.width, self.height
        w_num, h_num = int(round(w/unit)), int(round(h/unit))
        n = w_num * h_num
        points = np.zeros((n, 2))
        for h in range(h_num):
            for w in range(w_num):
                index, x, y = h * w_num + w, unit*w, unit*h
                points[index][:] = x, y
        points = [p*self.foot_to_meter for p in points]
        return points  

if __name__ == "__main__":
    model = BasketballCourt()
    points, line_index = model.wireframe()    
    grid_points = model.gridpoint(4)
    sio.savemat('basketball_model.mat', {'points':points, 'line_segment_index':line_index,
    'grid_points':grid_points})



       



        
    
    












    
