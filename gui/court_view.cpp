//
//  court_view.cpp
//  CameraCalibration
// 
//  Created by jimmy on 2019-02-02.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "court_view.hpp"       

CourtView::CourtView(std::string name) : CVImageView(name)
{

}

CourtView::~CourtView() {}

vector<vgl_point_2d<double>> CourtView::getPoints() const
{
	vector<vgl_point_2d<double>> points;
	for (auto iter = imagePoints.begin(); iter != imagePoints.end(); iter++)
	{
		int x = iter->x - 70;
		int y = 1090 - iter->y;

		double actual_x = 1.0 * x / 2400 * 60.96;
		double actual_y = 1.0 * y / 1020 * 25.908;

		vgl_point_2d<double> p(actual_x, actual_y);


		points.push_back(p);
	}

	return points;

}

void CourtView::annotate()
{
	cv::Point imagePos = getImagePosition();
	cv::Size imageSize = getImageSize();
	int activeArea = cvui::iarea(imagePos.x, imagePos.y, imageSize.width, imageSize.height);

	switch (activeArea)
	{
	case cvui::CLICK:
		cv::Point windowPoint = cvui::mouse();
		cv::Point imagePoint = imagePointForWindowPoint(windowPoint);

		windowPoints.push_back(windowPoint);
		imagePoints.push_back(imagePoint);
		break;
	}
}