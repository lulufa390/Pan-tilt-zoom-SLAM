#ifndef _ANNOTATION_WINDOW_
#define _ANNOTATION_WINDOW_

#include <iostream>
#include <cmath>

//#include <commdlg.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//#include <mat.h>

#include "cvimage_view.hpp"
#include "feature_annotation_view.hpp"
#include "court_view.hpp"
#include "optimization/camera_estimation.hpp"

class AnnotationWindow {

private:
	// pointer for two views
	FeatureAnnotationView * feature_annotation_view_;
	CourtView * court_view_;

private:
	// frame for main control view
	Mat frame_;
	cv::String main_view_name_;
	void mainControlHandler();
	void calibButtonFunc();
	void clearButtonFunc();

public:
	// interface for annotation application
	AnnotationWindow(cv::String name);
	~AnnotationWindow();

	void setFeatureAnnotationView(FeatureAnnotationView* image_view);

	void setCourtView(CourtView* image_view);

	// main function to start window application
	void startLoop();
};

#endif
