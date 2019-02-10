#define CVUI_IMPLEMENTATION
#include "annotation_window.h"

AnnotationWindow::AnnotationWindow(cv::String name)
{
	mainViewName = name;

	frame = cv::Mat(200, 400, CV_8UC3);
	frame = cv::Scalar(50, 50, 50);
}

AnnotationWindow::~AnnotationWindow() {}

void AnnotationWindow::setFeatureAnnotationView(FeatureAnnotationView* imageView)
{
	featureAnnotationView = imageView;
}

void AnnotationWindow::setCourtView(CourtView* imageView)
{
	courtView = imageView;
}

void AnnotationWindow::calibButtonFunc()
{
	vector<vgl_point_2d<double>> pointsAnno = featureAnnotationView->getPoints();
	vector<vgl_point_2d<double>> pointsCourt = courtView->getPoints();

	printf("Annotations in image:\n");
	for (auto iter = pointsAnno.begin(); iter != pointsAnno.end(); ++iter)
	{
		printf("(%f, %f)\n", iter->x(), iter->y());
	}

	printf("points in world coordinates:\n");
	for (auto iter = pointsCourt.begin(); iter != pointsCourt.end(); ++iter)
	{
		printf("(%f, %f)\n", iter->x(), iter->y());
	}

	vgl_point_2d<double> principal_point(640, 360);
	vpgl_perspective_camera<double> camera;

	cvx::init_calib(pointsCourt, pointsAnno, principal_point, camera);

}


void AnnotationWindow::clearButtonFunc()
{
	featureAnnotationView->clearAnnotations();
	courtView->clearAnnotations();
}

void AnnotationWindow::mainControlHandler()
{
	if (cvui::button(frame, 100, 40, "Do Calibration")) {
		calibButtonFunc();
	}

	if (cvui::button(frame, 100, 80, "Clear Annotations"))
	{
		clearButtonFunc();
	}
}

void AnnotationWindow::StartLoop() {

	const int viewNumber = 3;

	const cv::String featureViewName = featureAnnotationView->getWindowName();
	const cv::String courtViewName = courtView->getWindowName();

	const cv::String viewNames[] =
	{
		featureViewName,
		courtViewName,
		mainViewName,
	};

	// init multiple windows
	cvui::init(viewNames, viewNumber);

	while (true) {


		cvui::context(featureViewName);
		featureAnnotationView->drawFrame();
		featureAnnotationView->annotate();
		cvui::imshow(featureViewName, featureAnnotationView->frame);

		cvui::context(courtViewName);
		courtView->drawFrame();
		courtView->annotate();
		cvui::imshow(courtViewName, courtView->frame);

		cvui::context(mainViewName);
		mainControlHandler();
		cvui::imshow(mainViewName, frame);


		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			break;
		}
	}
}

