#define CVUI_IMPLEMENTATION
#include "annotation_window.h"

AnnotationWindow::AnnotationWindow(cv::String name)
{
	mainViewName = name;

	frame = cv::Mat(200, 400, CV_8UC3);
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

void AnnotationWindow::mainControlHandler()
{
	if (cvui::button(frame, 100, 40, "Do Calibration")) {
		vector<vgl_point_2d<double>> pointsAnno = featureAnnotationView->getPoints();

		vector<vgl_point_2d<double>> pointsCourt = courtView->getPoints();

		vector<cv::Point> cvPointAnno;
		vector<cv::Point> cvPointCourt;


		for (auto iter = pointsAnno.begin(); iter != pointsAnno.end(); ++iter)
		{
			printf("Annotations: (%f, %f)\n", iter->x(), iter->y());

			cvPointAnno.push_back(cv::Point(iter->x(), iter->y()));
		}

		for (auto iter = pointsCourt.begin(); iter != pointsCourt.end(); ++iter)
		{
			printf("Court: (%f, %f)\n", iter->x(), iter->y());

			cvPointCourt.push_back(cv::Point(iter->x(), iter->y()));
		}

		vgl_point_2d<double> principal_point(640, 360);
		vpgl_perspective_camera<double> camera;

		cvx::init_calib(pointsCourt, pointsAnno, principal_point, camera);

		printf("here debug!");

		/*cv::Mat homographyMat = cv::findHomography(cvPointCourt, cvPointAnno);

		printf("vector size: %d, %d", cvPointCourt.size(), cvPointAnno.size());

		printf("size: %d, %d", homographyMat.size().height, homographyMat.size().width);

		for (int i = 0; i < homographyMat.size().height; ++i)
		{
		for (int j = 0; j < homographyMat.size().width; ++j)
		{
		printf("%f ", homographyMat.at<float>(i, j));
		}
		printf("\n");
		}*/
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

