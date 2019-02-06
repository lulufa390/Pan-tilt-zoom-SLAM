#ifndef _ANNOTATION_WINDOW_
#define _ANNOTATION_WINDOW_

#include <iostream>
#include <cmath>

//#include <commdlg.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mat.h>

#include "cvui.h"
#include "cvimage_view.hpp"

//typedef void(*MouseClickCallback)(cv::Point);
//typedef cv::Mat(*ButtonClickCallback)(cv::Mat img, std::vector<cv::Point> source_img_pts, std::vector<cv::Point2d> model_img_pts);

class AnnotationWindow{

private:
	std::vector<CVImageView> image_views;

public:

	AnnotationWindow();
	~AnnotationWindow();

	void addImageView(CVImageView & newImageView);
	void clearImageViews();

	
	void StartLoop();

	//void set_source_img(cv::Mat & origin_img);
	//void set_model_img(cv::Mat & model_img);

	//void set_button_click_event(ButtonClickCallback buttonClick);

	//void draw_point(cv::Point p);

	//// coordinates transformation between model image and actual model points.
	//cv::Point2d hockey_transformation(cv::Point p);

//public:
//	// window
//	std::string window_name;
//	cv::Mat frame;
//
//	// images to show
//	cv::Mat source_img;
//	cv::Mat model_img;
//	cv::Mat visualize_img;
//
//	// position and size of showed images
//	const cv::Size img_size = cv::Size(640, 360);
//	const cv::Point source_img_position = cv::Point(20, 10);
//	const cv::Point model_img_position = cv::Point(680, 10);
//	const cv::Point visualize_img_position = cv::Point(680, 380);
//
//	// vector of points to show
//	std::vector<cv::Point> source_img_pts_show;
//	std::vector<cv::Point> model_img_pts_show;
//
//	// vector of points in origin images
//	std::vector<cv::Point> source_img_pts;
//	std::vector<cv::Point2d> model_img_pts;
//
//	// origin size of images
//	cv::Mat origin_source_img;
//	cv::Mat origin_model_img;

	//cv::Size source_img_origin_size;
	//cv::Size model_img_origin_size;


protected:
	//MouseClickCallback mouse_click;
	//ButtonClickCallback calib_button;

	
};

#endif