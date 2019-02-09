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
#include "optimization\camera_estimation.hpp"

class AnnotationWindow {

private:
	FeatureAnnotationView * featureAnnotationView;
	CourtView * courtView;

	// frame for main control view
	Mat frame;
	cv::String mainViewName;
	void mainControlHandler();


public:

	AnnotationWindow(cv::String name);
	~AnnotationWindow();

	void setFeatureAnnotationView(FeatureAnnotationView* imageView);

	void setCourtView(CourtView* imageView);

	// main function to start window application
	void StartLoop();
};

#endif