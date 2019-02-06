#define CVUI_IMPLEMENTATION
#include "annotation_window.h"

AnnotationWindow::AnnotationWindow()
{
	//// base information of the window
	//frame = cv::Mat(750, 1400, CV_8UC3);
	//frame = cv::Scalar(50, 50, 50);
	//window_name = "Camera Calibration Tool";

	//// the source image and model image are set to be black as default
	//source_img = cv::Mat(img_size.height, img_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
	//model_img = cv::Mat(img_size.height, img_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
	//visualize_img = cv::Mat(img_size.height, img_size.width, CV_8UC3, cv::Scalar(0, 0, 0));

}

AnnotationWindow::~AnnotationWindow(){}


void AnnotationWindow::addImageView(CVImageView & newImageView)
{
	image_views.push_back(newImageView);
}

void AnnotationWindow::clearImageViews()
{
	image_views.clear();
}

//void AnnotationWindow::set_source_img(cv::Mat & img)
//{
//	origin_source_img = img;
//	//source_img_origin_size = cv::Size(origin_source_img.cols, origin_source_img.rows);
//	cv::resize(origin_source_img, source_img, img_size, cv::INTER_LINEAR);
//}

//void AnnotationWindow::set_model_img(cv::Mat & img)
//{
//	origin_model_img = img;
//	//model_img_origin_size = cv::Size(origin_model_img.cols, origin_model_img.rows);
//	cv::resize(origin_model_img, model_img, img_size, cv::INTER_LINEAR);
//}

//void AnnotationWindow::set_button_click_event(ButtonClickCallback buttonClick)
//{
//	calib_button = buttonClick;
//}

//void AnnotationWindow::draw_point(cv::Point p)
//{
//	cv::Point leftup = cv::Point(p.x - 5, p.y - 5);
//	cv::Point leftdown = cv::Point(p.x - 5, p.y + 5);
//	cv::Point rightup = cv::Point(p.x + 5, p.y - 5);
//	cv::Point rightdown = cv::Point(p.x + 5, p.y + 5);
//	cv::line(frame, leftup, rightdown, cv::Scalar(255, 0, 0), 1);
//	cv::line(frame, leftdown, rightup, cv::Scalar(255, 0, 0), 1);
//}

void AnnotationWindow::StartLoop(){

	int windowCnt = image_views.size();
	cv::String * windowArray = new cv::String[windowCnt];
	for (int i = 0; i < windowCnt; i++)
	{
		windowArray[i] = image_views[i].getWindowName();
	}

	// init multiple windows
	cvui::init(windowArray, windowCnt);

	while (true) {

		for (int i = 0; i < windowCnt; i++)
		{
			CVImageView & view = image_views[i];
			cv::String name = view.getWindowName();

			cvui::context(name);
			
			cvui::imshow(name, view.getFrame());
			// mainloop for each window



		}


		//cvui::image(frame, source_img_position.x, source_img_position.y, source_img);
		//cvui::image(frame, model_img_position.x, model_img_position.y, model_img);
		//cvui::image(frame, visualize_img_position.x, visualize_img_position.y, visualize_img);

		//int source_img_status = cvui::iarea(source_img_position.x, source_img_position.y, img_size.width, img_size.height);
		//int model_img_status = cvui::iarea(model_img_position.x, model_img_position.y, img_size.width, img_size.height);

		//switch (source_img_status)
		//{
		//	case cvui::CLICK:	
		//		
		//		double scale_x = 1.0 * (origin_source_img.size().width - 1) / (img_size.width - 1);
		//		double scale_y = 1.0 * (origin_source_img.size().height - 1) / (img_size.height - 1);
		//		int x = round((cvui::mouse().x - source_img_position.x) * scale_x);
		//		int y = round((cvui::mouse().y - source_img_position.y) * scale_y);

		//		source_img_pts.push_back(cv::Point(x, y));
		//		source_img_pts_show.push_back(cvui::mouse());

		//		//printf("source img: (%d, %d)\n", x, y);

		//		break;
		//}

		//switch (model_img_status)
		//{
		//	case cvui::CLICK:
		//		
		//		double scale_x = 1.0 * (origin_model_img.size().width - 1) / (img_size.width - 1);
		//		double scale_y = 1.0 * (origin_model_img.size().height - 1) / (img_size.height - 1);

		//		//printf("scale: %f, %f\n", scale_x, scale_y);

		//		int x = round((cvui::mouse().x - model_img_position.x) * scale_x);
		//		int y = round((cvui::mouse().y - model_img_position.y) * scale_y);

		//		cv::Point2d actual_point = hockey_transformation(cv::Point(x, y));

		//		model_img_pts.push_back(actual_point);
		//		model_img_pts_show.push_back(cvui::mouse());


		//		//printf("model img before: (%d, %d)\n", x, y);
		//		//printf("model img: (%f, %f)\n", actual_point.x, actual_point.y);
		//		break;
		//}

		//for (auto iter = source_img_pts_show.begin(); iter != source_img_pts_show.end(); iter++)
		//{
		//	draw_point(*iter);
		//}

		//for (auto iter = model_img_pts_show.begin(); iter != model_img_pts_show.end(); iter++)
		//{
		//	draw_point(*iter);
		//}


		//if (cvui::button(frame, 20, 500, 100, 30, "Start Calibration")) {

		//	cv::Mat origin_visualize = calib_button(origin_source_img, source_img_pts, model_img_pts);

		//	cv::resize(origin_visualize, visualize_img, img_size, cv::INTER_LINEAR);

		//	//// button was clicked
		//	//OPENFILENAME ofn;
		//	//::memset(&ofn, 0, sizeof(ofn));
		//	//char f1[100];
		//	//f1[0] = 0;
		//	//ofn.lStructSize = sizeof(ofn);

		//	//

		//	//ofn.lpstrTitle = "Select A File";
		//	//ofn.lpstrFilter = "Text Files\0*.txt\0All Files\0*.*\0\0";
		//	//ofn.nFilterIndex = 2;
		//	//ofn.lpstrFile = f1;
		//	//ofn.nMaxFile = 100;
		//	//ofn.Flags = OFN_FILEMUSTEXIST;

		//	//if (::GetOpenFileName(&ofn) != FALSE)
		//	//{
		//	//	// ofn.lpstrFile will have the full path and file name.
		//	//	// For example:  std::cout << ofn.lpstrFile << std::endl
		//	//	//   #include <iostream> to try last line - prints file name to console.
		//	//	//  You can allocate another buffer and put it in ofn.lpstrFileTitle and it
		//	//	//  will return the file anme only.
		//	//}
		//}

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		//cvui::update();

		// Show everything on the screen
		//cv::imshow(window_name, frame);



		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			break;
		}
	}

}

//cv::Point2d AnnotationWindow::hockey_transformation(cv::Point p)
//{
//	int x = p.x - 70;
//	int y = 1090 - p.y;
//
//	
//	double actual_x = 1.0 * x / 2400 * 60.96;
//	double actual_y = 1.0 * y / 1020 * 25.908;
//
//	return cv::Point2d(actual_x, actual_y);
//}