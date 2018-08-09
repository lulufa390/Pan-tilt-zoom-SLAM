from synthesize import *

pts = generate_points(3000)

"""          
load the soccer field model
"""

soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")
line_index = soccer_model['line_segment_index']
points = soccer_model['points']

"""  
load the sequence annotation     
"""

seq = sio.loadmat("./two_point_calib_dataset/highlights/seq3_anno.mat")
annotation = seq["annotation"]
meta = seq['meta']
proj_center = meta[0][0]["cc"][0]

base_rotation = np.zeros([3, 3])
cv.Rodrigues(meta[0][0]["base_rotation"][0], base_rotation)


draw_3d_model(line_index, points, pts)

""" 
compute the rays of these synthesized points
"""
rays = []
for i in range(0, len(pts)):
    ray = compute_rays(proj_center, pts[i], base_rotation)
    rays.append(ray)

""" 
This part is used to show the synthesized images  
"""

for i in range(annotation.size):

    img = np.zeros((720, 1280, 3), np.uint8)
    img.fill(255)

    u, v, f = annotation[0][i]['camera'][0][0:3]
    pan, tilt, _ = annotation[0][i]['ptz'].squeeze() * pi / 180

    draw_soccer_line(img, u, v, f, pan, tilt, base_rotation, proj_center, line_index, points)

    # draw the feature points in images
    for j in range(len(pts)):
        p = np.array(pts[j])

        res = from_3d_to_2d(u, v, f, pan, tilt, proj_center, base_rotation, p)
        res2 = from_pan_tilt_to_2d(u, v, f, pan, tilt, rays[j][0], rays[j][1])

        if 0 < res[0] < 1280 and 0 < res[1] < 720:
            print(p)
            print("ray", rays[j][0] * 180 / pi, rays[j][1] * 180 / pi)
            print("res:, ", res)
            print("res2: ", res2)
            print("==========")

        cv.circle(img, (int(res[0]), int(res[1])), color=(0, 0, 0), radius=8, thickness=2)
        cv.circle(img, (int(res2[0]), int(res2[1])), color=(255, 0, 0), radius=8, thickness=2)

    cv.imshow("synthesized image", img)
    cv.waitKey(0)

"""
This part saves synthesized data to mat file
"""
save_to_mat_degree(pts, rays)
