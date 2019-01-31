#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "annotation_window.h"

void callback(cv::Point p){
	printf("%d, %d", p.x, p.y);
}


cv::Mat calib_button(cv::Mat img, std::vector<cv::Point> source_img_pts, std::vector<cv::Point2d> model_img_pts)
{
	for (auto iter = source_img_pts.begin(); iter != source_img_pts.end(); iter++)
	{
		printf("(%d, %d). \n", iter->x, iter->y);
	}

	for (auto iter = model_img_pts.begin(); iter != model_img_pts.end(); iter++)
	{
		printf("(%f, %f). \n", iter->x, iter->y);
	}


	cv::Mat homography_mat = cv::findHomography(model_img_pts, source_img_pts);

	printf("%d, %d", homography_mat.size().width, homography_mat.size().height);

	return img;
}


int main(int argc, char *argv[])
{
	cv::Mat source_img = cv::imread("1.jpg", cv::IMREAD_COLOR);
	cv::Mat model_img = cv::imread("model.png", cv::IMREAD_COLOR);


	AnnotationWindow window;

	window.set_source_img(source_img);
	window.set_model_img(model_img);

	window.set_button_click_event(calib_button);

	window.StartLoop();
	return 0;
}