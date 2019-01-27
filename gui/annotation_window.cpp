#define CVUI_IMPLEMENTATION
#include "annotation_window.h"

AnnotationWindow::AnnotationWindow(MouseClickCallback mouseClick){
	// base information of the window
	frame = cv::Mat(750, 1400, CV_8UC3);
	frame = cv::Scalar(50, 50, 50);
	window_name = "Camera Calibration Tool";

	// the source image and model image are set to be black as default
	source_img = cv::Mat(img_size.height, img_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
	model_img = cv::Mat(img_size.height, img_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
}

AnnotationWindow::~AnnotationWindow(){}


void AnnotationWindow::set_source_img(cv::Mat & origin_source_img)
{
	source_img_origin_size = cv::Size(origin_source_img.cols, origin_source_img.rows);
	cv::resize(origin_source_img, source_img, img_size, cv::INTER_LINEAR);
}

void AnnotationWindow::set_model_img(cv::Mat & origin_model_img)
{
	model_img_origin_size = cv::Size(origin_model_img.cols, origin_model_img.rows);
	cv::resize(origin_model_img, model_img, img_size, cv::INTER_LINEAR);
}

void AnnotationWindow::draw_point(cv::Point p)
{
	cv::Point leftup = cv::Point(p.x - 5, p.y - 5);
	cv::Point leftdown = cv::Point(p.x - 5, p.y + 5);
	cv::Point rightup = cv::Point(p.x + 5, p.y - 5);
	cv::Point rightdown = cv::Point(p.x + 5, p.y + 5);
	cv::line(frame, leftup, rightdown, cv::Scalar(255, 0, 0), 1);
	cv::line(frame, leftdown, rightup, cv::Scalar(255, 0, 0), 1);
}

void AnnotationWindow::StartLoop(){

	cvui::init(window_name);

	while (true) {

		cvui::image(frame, source_img_position.x, source_img_position.y, source_img);
		cvui::image(frame, model_img_position.x, model_img_position.y, model_img);

		int source_img_status = cvui::iarea(source_img_position.x, source_img_position.y, img_size.width, img_size.height);
		int model_img_status = cvui::iarea(model_img_position.x, model_img_position.y, img_size.width, img_size.height);

		switch (source_img_status)
		{
			case cvui::CLICK:	
				
				double scale_x = 1.0 * (source_img_origin_size.width - 1) / (img_size.width - 1);
				double scale_y = 1.0 * (source_img_origin_size.height - 1) / (img_size.height - 1);
				int x = round((cvui::mouse().x - source_img_position.x) * scale_x);
				int y = round((cvui::mouse().y - source_img_position.y) * scale_y);

				source_img_pts.push_back(cv::Point(x, y));
				source_img_pts_show.push_back(cvui::mouse());

				break;
		}

		switch (model_img_status)
		{
			case cvui::CLICK:
				
				double scale_x = 1.0 * (model_img_origin_size.width - 1) / (img_size.width - 1);
				double scale_y = 1.0 * (model_img_origin_size.height - 1) / (img_size.height - 1);
				int x = round((cvui::mouse().x - model_img_position.x) * scale_x);
				int y = round((cvui::mouse().y - model_img_position.y) * scale_y);

				model_img_pts.push_back(cv::Point(x, y));
				model_img_pts_show.push_back(cvui::mouse());

				
				break;
		}

		for (auto iter = source_img_pts_show.begin(); iter != source_img_pts_show.end(); iter++)
		{
			draw_point(*iter);
		}

		for (auto iter = model_img_pts_show.begin(); iter != model_img_pts_show.end(); iter++)
		{
			draw_point(*iter);
		}


		if (cvui::button(frame, 20, 500, 100, 30, "Button")) {
			// button was clicked
		}

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