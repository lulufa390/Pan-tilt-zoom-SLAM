//
//  feature_annotation_view.cpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-02.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "feature_annotation_view.hpp"

FeatureAnnotationView::FeatureAnnotationView(std::string name) : CVImageView(name)
{

}

FeatureAnnotationView::~FeatureAnnotationView() {}

// get points on image coordinate
vector<vgl_point_2d<double>> FeatureAnnotationView::getPoints() const
{
	vector<vgl_point_2d<double>> points;
	for (auto iter = imagePoints.begin(); iter != imagePoints.end(); iter++)
	{
		vgl_point_2d<double> p((double)iter->x, (double)iter->y);
		points.push_back(p);
	}

	return points;
}

void FeatureAnnotationView::annotate()
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
