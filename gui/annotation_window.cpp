#define CVUI_IMPLEMENTATION
#include "annotation_window.h"

AnnotationWindow::AnnotationWindow(MouseClickCallback mouseClick){
	frame = cv::Mat(750, 1400, CV_8UC3);
	frame = cv::Scalar(50, 50, 50);
	window_name = "Camera Calibration Tool";

	source_img = cv::Mat(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));
	model_img = cv::Mat(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));
}

AnnotationWindow::~AnnotationWindow(){}


void AnnotationWindow::set_source_img(cv::Mat & origin_source_img)
{
	cv::resize(origin_source_img, source_img, cv::Size(640, 360), cv::INTER_LINEAR);
}

void AnnotationWindow::set_model_img(cv::Mat & origin_model_img)
{
	cv::resize(origin_model_img, model_img, cv::Size(640, 360), cv::INTER_LINEAR);
}

void AnnotationWindow::StartLoop(){

	cvui::init(window_name);

	while (true) {

		cvui::image(frame, 20, 20, source_img);

		cvui::image(frame, 680, 20, model_img);


		// Check what is the current status of the mouse cursor
		// regarding the previously rendered rectangle.
		int status = cvui::iarea(20, 20, 640, 360);

		switch (status)
		{
			case cvui::CLICK:	std::cout << "Rectangle was clicked!" << std::endl; break;
			case cvui::DOWN:	cvui::printf(frame, 240, 70, "Mouse is: DOWN"); break;
			case cvui::OVER:	cvui::printf(frame, 240, 70, "Mouse is: OVER"); break;
			case cvui::OUT:		cvui::printf(frame, 240, 70, "Mouse is: OUT"); break;
		}

		// Show the coordinates of the mouse pointer on the screen
		cvui::printf(frame, 240, 500, "Mouse pointer is at (%d,%d)", cvui::mouse().x, cvui::mouse().y);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();

		// Show everything on the screen
		cv::imshow(window_name, frame);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			break;
		}
	}

}