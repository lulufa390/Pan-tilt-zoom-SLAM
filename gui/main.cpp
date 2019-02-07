#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vpgl/vpgl_perspective_camera.h>
#include <vgl/vgl_point_2d.h>

#include "annotation_window.h"

void callback(cv::Point p) {
	printf("%d, %d", p.x, p.y);
}


cv::Mat calib_button(cv::Mat img, std::vector<cv::Point> source_img_pts, std::vector<cv::Point2d> model_img_pts)
{

	MATFile *pmatFile = NULL;
	mxArray *pMxArray = NULL;

	// 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
	double *initA;

	pmatFile = matOpen("ice_hockey_model.mat", "r");
	pMxArray = matGetVariable(pmatFile, "points");





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
	std::vector<vgl_point_2d<double>> points;

	AnnotationWindow app;

	FeatureAnnotationView featureAnnotation("Feature Annotation");
	CourtView courtView("Court View");

	cv::Mat source_img = cv::imread("C:/graduate_design/cvui/cvuiApp/cvuiApp/1.jpg", cv::IMREAD_COLOR);
	cv::Mat model_img = cv::imread("C:/graduate_design/cvui/cvuiApp/cvuiApp/model.png", cv::IMREAD_COLOR);

	featureAnnotation.setImage(source_img);
	courtView.setImage(model_img);

	app.addImageView(&featureAnnotation);
	app.addImageView(&courtView);
	app.StartLoop();

	//cv::imshow("fuck", source_img);

	//AnnotationWindow window;

	//window.set_source_img(source_img);
	//window.set_model_img(model_img);

	//window.set_button_click_event(calib_button);
	//window.StartLoop();

	return 0;
}