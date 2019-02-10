#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vpgl/vpgl_perspective_camera.h>
#include <vgl/vgl_point_2d.h>

#include "annotation_window.h"

using std::string;

int main(int argc, char *argv[])
{
#ifdef _WIN32
    string im_name = string("C:/graduate_design/cvui/cvuiApp/cvuiApp/1.jpg");
    string model_name = string("C:/graduate_design/cvui/cvuiApp/cvuiApp/model.png");
#elif __APPLE__
    string im_name("/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/gui/1.jpg");
    string model_name("/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/gui/model.png");
#endif
	std::vector<vgl_point_2d<double>> points;

	AnnotationWindow app("Main View");

	FeatureAnnotationView featureAnnotation("Feature Annotation");
	CourtView courtView("Court View");

	courtView.setWindowSize(cv::Size(800, 400));
	courtView.setImagePosition(cv::Point(30, 20));
	courtView.setImageSize(cv::Size(762, 348));

	cv::Mat source_img = cv::imread(im_name, cv::IMREAD_COLOR);
	cv::Mat model_img = cv::imread(model_name, cv::IMREAD_COLOR);

	featureAnnotation.setImage(source_img);
	courtView.setImage(model_img);

	app.setFeatureAnnotationView(&featureAnnotation);
	app.setCourtView(&courtView);
	app.StartLoop();

	return 0;
}
