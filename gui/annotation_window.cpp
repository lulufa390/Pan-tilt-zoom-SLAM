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

AnnotationWindow::~AnnotationWindow() {}


void AnnotationWindow::addImageView(CVImageView * newImageView)
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

void AnnotationWindow::StartLoop() {

	int windowCnt = image_views.size();
	cv::String * windowArray = new cv::String[windowCnt];
	for (int i = 0; i < windowCnt; i++)
	{
		windowArray[i] = image_views[i]->getWindowName();
	}

	// init multiple windows
	cvui::init(windowArray, windowCnt);

	while (true) {

		for (int i = 0; i < windowCnt; i++)
		{
			CVImageView * view = image_views[i];
			cv::String name = view->getWindowName();

			cvui::context(name);



			// mainloop for each window
			view->drawFrame();
			view->annotate();


			cvui::imshow(name, view->frame);

		}

		if (cv::waitKey(20) == 'P') {

			FeatureAnnotationView * viewAnno = (FeatureAnnotationView *)image_views[0];

			CourtView * viewCourt = (CourtView *)image_views[1];

			vector<vgl_point_2d<double>> pointsAnno = viewAnno->getPoints();

			vector<vgl_point_2d<double>> pointsCourt = viewCourt->getPoints();

			for (auto iter = pointsAnno.begin(); iter != pointsAnno.end(); ++iter)
			{
				printf("Annotations: (%f, %f)\n", iter->x(), iter->y());
			}

			for (auto iter = pointsCourt.begin(); iter != pointsCourt.end(); ++iter)
			{
				printf("Court: (%f, %f)\n", iter->x(), iter->y());
			}


		}

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