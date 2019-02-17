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
	//@todo replace this code with function in play_filed_
	//vector<vgl_point_2d<double>> points;
	//for (auto iter = imagePoints.begin(); iter != imagePoints.end(); iter++)
	//{
	//	vgl_point_2d<double> p;

	//	bool is_find = play_field_->find_candinate_point(*iter, p, 10);

	//	if (is_find)
	//	{

	//	}

	//	//int x = iter->x - 70;
	//	//int y = 1090 - iter->y;

	//	//double actual_x = 1.0 * x / 2400 * 60.96;
	//	//double actual_y = 1.0 * y / 1020 * 25.908;

	//	//vgl_point_2d<double> p(actual_x, actual_y);


	//	points.push_back(p);
	//}

	return world_points_;

}

void CourtView::annotate()
{
	cv::Point image_pos = getImagePosition();
	cv::Size image_size = getImageSize();
	int active_area = cvui::iarea(image_pos.x, image_pos.y, image_size.width, image_size.height);

	switch (active_area)
	{
	case cvui::CLICK:
		cv::Point window_point = cvui::mouse();
		cv::Point image_point = imagePointForWindowPoint(window_point);

		vgl_point_2d<double> image_point_vgl = vgl_point_2d<double>(image_point.x, image_point.y);
		vgl_point_2d<double> world_point;

		bool is_find = play_field_->find_candinate_point(image_point_vgl, world_point, 10);
		if (is_find) {
			vgl_point_2d<double> reverse_image_point = play_field_->world_point_to_image_point(world_point);
			cv::Point reverse_image_point_cv = cv::Point(reverse_image_point.x(), reverse_image_point.y());
			cv::Point reverse_window_point_cv = windowPointForImagePoint(reverse_image_point_cv);

			windows_points_.push_back(reverse_window_point_cv);
			//image_points_.push_back(image_point_vgl);

			world_points_.push_back(world_point);
		}

		break;
	}
}

void CourtView::clearAnnotations()
{
	windows_points_.clear();
	//image_points_.clear();
	world_points_.clear();
}
