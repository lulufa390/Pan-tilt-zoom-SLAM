#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vpgl/vpgl_perspective_camera.h>
#include <vgl/vgl_point_2d.h>

#include "annotation_window.h"


int main(int argc, char *argv[])
{
	std::vector<vgl_point_2d<double>> points;

	AnnotationWindow app("Main View");

	FeatureAnnotationView featureAnnotation("Feature Annotation");
	CourtView courtView("Court View");

	cv::Mat source_img = cv::imread("C:/graduate_design/cvui/cvuiApp/cvuiApp/1.jpg", cv::IMREAD_COLOR);
	cv::Mat model_img = cv::imread("C:/graduate_design/cvui/cvuiApp/cvuiApp/model.png", cv::IMREAD_COLOR);

	featureAnnotation.setImage(source_img);
	courtView.setImage(model_img);

	app.setFeatureAnnotationView(&featureAnnotation);
	app.setCourtView(&courtView);
	app.StartLoop();

	return 0;
}