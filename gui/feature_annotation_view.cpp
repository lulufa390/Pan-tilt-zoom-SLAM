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

// get line segment
vector<vgl_line_segment_2d<double>> FeatureAnnotationView::getLines() const
{
	return lines_;
}

// get image point that are on circles
vector<vgl_point_2d<double>> FeatureAnnotationView::getCirclePoints() const
{
	return circle_points_;
}

void FeatureAnnotationView::annotate()
{
	cv::Point image_pos = getImagePosition();
	cv::Size image_size = getImageSize();
	int active_area = cvui::iarea(image_pos.x, image_pos.y, image_size.width, image_size.height);

	// for drawing line, save the begin point, and whether is continue a line
	static bool continue_line = false;
	static cv::Point line_begin;

	switch (active_area)
	{
	case cvui::CLICK:
		cv::Point window_point = cvui::mouse();
		cv::Point image_point = imagePointForWindowPoint(window_point);

		if (state_ == AnnotationState::point)
		{
			windows_points_.push_back(window_point);
			image_points_.push_back(image_point);
			continue_line = false;
		}
		else if (state_ == AnnotationState::line)
		{
			if (continue_line)
			{
				cv::Point vgl_begin = imagePointForWindowPoint(line_begin);
				vgl_point_2d<double> p1(vgl_begin.x, vgl_begin.y);

				cv::Point line_end = cvui::mouse();
				cv::Point vgl_end = imagePointForWindowPoint(line_end);
				vgl_point_2d<double> p2(vgl_end.x, vgl_end.y);

				vgl_line_segment_2d<double> line(p1, p2);

				windows_line_.push_back(std::pair<cv::Point, cv::Point>(line_begin, line_end));

				lines_.push_back(line);

				continue_line = false;
			}
			else
			{
				line_begin = cvui::mouse();
				continue_line = true;
			}
		}
		else
		{
			windows_points_circle_.push_back(window_point);
			vgl_point_2d<double> circle(image_point.x, image_point.y);
			circle_points_.push_back(circle);
			continue_line = false;
		}


		break;
	}
}

void FeatureAnnotationView::clearAnnotations()
{
	windows_points_.clear();
	image_points_.clear();

	windows_points_circle_.clear();
	circle_points_.clear();

	windows_line_.clear();
	lines_.clear();
}
