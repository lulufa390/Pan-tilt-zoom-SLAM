import cv2 as cv
from image_process import *


def video_capture(file_path, save_path, begin_time, rate, length):
    video = cv.VideoCapture(file_path)

    for i in range(0, length):
        print(i)
        video.set(cv.CAP_PROP_POS_MSEC, begin_time + i / rate * 1000)
        _, img = video.read()

        # img = blur_sub_image(img, 110, 55, 120, 120)

        cv.imshow("test", img)

        cv.imwrite(save_path + "/" + str(i + 1) + ".jpg", img)

        cv.waitKey(0)


# video_capture = cv.VideoCapture("../../dataset/Charlotte_Eagles_8_2_2014.mp4")

# video_capture = cv.VideoCapture("../../dataset/HIGHLIGHTS Orlando City vs Pittsburgh Riverhounds _ 3_29_2014.mp4")

if __name__ == '__main__':
    video_capture("/hdd/luke/hockey_data/USA 2-3 Canada - Men's Ice Hockey Gold Medal Match _ Vancouver 2010 Winter Olympics.mp4",
                  "/hdd/luke/hockey_data/Olympic/", 4072000, 25, 625)

