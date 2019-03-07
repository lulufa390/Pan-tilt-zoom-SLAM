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
	play_field_ = std::make_shared<NHLIceHockeyPlayField>();
}

CourtView::~CourtView() {}

vector<vgl_point_2d<double>> CourtView::getPoints() const
{
	return world_points_;
}

vector<vgl_line_segment_2d<double>> CourtView::getLines() const
{
	return lines_;
}

void CourtView::annotate()
{
	cv::Point image_pos = getImagePosition();
	cv::Size image_size = getImageSize();
	int active_area = cvui::iarea(image_pos.x, image_pos.y, image_size.width, image_size.height);

	// for drawing line, save the begin point, and whether is continue a line
	static bool continue_line = false;
	static cv::Point line_begin;
	static vgl_point_2d<double> line_being_world;

	switch (active_area)
	{
	case cvui::CLICK:
		cv::Point window_point = cvui::mouse();
		cv::Point image_point = imagePointForWindowPoint(window_point);

		vgl_point_2d<double> image_point_vgl = vgl_point_2d<double>(image_point.x, image_point.y);
		vgl_point_2d<double> world_point_feet;

		bool is_find = play_field_->find_candinate_point(image_point_vgl, world_point_feet, 15);
		if (is_find) {
			vgl_point_2d<double> reverse_image_point = play_field_->world_point_to_image_point(world_point_feet);
			cv::Point reverse_image_point_cv = cv::Point(reverse_image_point.x(), reverse_image_point.y());
			cv::Point reverse_window_point_cv = windowPointForImagePoint(reverse_image_point_cv);

			vgl_point_2d<double> world_point_meter(world_point_feet.x() * 0.3048, world_point_feet.y() * 0.3048);

			if (state_ == AnnotationState::point)
			{
				windows_points_.push_back(reverse_window_point_cv);

				world_points_.push_back(world_point_meter);
				continue_line = false;
			}
			else if (state_ == AnnotationState::line)
			{
				if (continue_line)
				{
					cv::Point line_end = reverse_window_point_cv;
					vgl_point_2d<double> line_end_world = world_point_meter;

					windows_line_.push_back(std::pair<cv::Point, cv::Point>(line_begin, line_end));

					lines_.push_back(vgl_line_segment_2d<double>(line_being_world, line_end_world));

					continue_line = false;
				}
				else
				{
					line_begin = reverse_window_point_cv;
					line_being_world = world_point_meter;
					continue_line = true;
				}
			}
			else
			{
				continue_line = false;
			}

		}

		break;
	}
}

void CourtView::clearAnnotations()
{
	windows_points_.clear();
	world_points_.clear();

	windows_line_.clear();
	lines_.clear();
}
