#ifndef _ANNOTATION_WINDOW_
#define _ANNOTATION_WINDOW_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cvui.h"

typedef void(*MouseClickCallback)(cv::Point);

class AnnotationWindow{

public:

	AnnotationWindow(MouseClickCallback mouseClick);

	void StartLoop();

	~AnnotationWindow();

	void set_source_img(cv::Mat & origin_img);
	void set_model_img(cv::Mat & model_img);


public:
	std::string window_name;
	cv::Mat frame;

	cv::Mat source_img;
	cv::Mat model_img;
	cv::Mat calib_result_img;

};

#endif