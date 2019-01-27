#ifndef _ANNOTATION_WINDOW_
#define _ANNOTATION_WINDOW_

#include <iostream>
#include <cmath>
#include <commdlg.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cvui.h"

typedef void(*MouseClickCallback)(cv::Point);
typedef void(*ButtonClickCallback)(std::vector<cv::Point> source_img_pts, std::vector<cv::Point> model_img_pts);

class AnnotationWindow{

public:

	AnnotationWindow(MouseClickCallback mouseClick);

	void StartLoop();

	~AnnotationWindow();

	void set_source_img(cv::Mat & origin_img);
	void set_model_img(cv::Mat & model_img);

	void draw_point(cv::Point p);


public:
	// window
	std::string window_name;
	cv::Mat frame;

	// images to show
	cv::Mat source_img;
	cv::Mat model_img;
	cv::Mat calib_result_img;

	// position and size of showed images
	const cv::Size img_size = cv::Size(640, 360);
	const cv::Point source_img_position = cv::Point(20, 20);
	const cv::Point model_img_position = cv::Point(680, 20);

	// vector of points to show
	std::vector<cv::Point> source_img_pts_show;
	std::vector<cv::Point> model_img_pts_show;

	// vector of points in origin images
	std::vector<cv::Point> source_img_pts;
	std::vector<cv::Point> model_img_pts;

	// origin size of images
	cv::Size source_img_origin_size;
	cv::Size model_img_origin_size;
};

#endif