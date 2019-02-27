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
	for (auto iter = image_points_.begin(); iter != image_points_.end(); iter++)
	{
		vgl_point_2d<double> p((double)iter->x, (double)iter->y);
		points.push_back(p);
	}

	return points;
}

void FeatureAnnotationView::annotate()
{
	cv::Point image_pos = getImagePosition();
	cv::Size image_size = getImageSize();
	int active_area = cvui::iarea(image_pos.x, image_pos.y, image_size.width, image_size.height);

	switch (active_area)
	{
	case cvui::CLICK:
		cv::Point window_point = cvui::mouse();
		cv::Point image_point = imagePointForWindowPoint(window_point);
            //std::cout<<"window point: "<< window_point<<std::endl;
            //std::cout<<"image point: "<<image_point<<std::endl;
		windows_points_.push_back(window_point);
		image_points_.push_back(image_point);
		break;
	}
}

void FeatureAnnotationView::clearAnnotations()
{
	windows_points_.clear();
	image_points_.clear();
}
