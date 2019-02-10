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

	return world_points;

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

		vgl_point_2d<double> image_point_vgl = vgl_point_2d<double>(imagePoint.x, imagePoint.y);
		vgl_point_2d<double> world_point;


		printf("(%f, %f)\n", image_point_vgl.x(), image_point_vgl.y());
		bool is_find = play_field_->find_candinate_point(image_point_vgl, world_point, 10);
		if (is_find) {
			printf("Has find\n");
			windowPoints.push_back(windowPoint);
			imagePoints.push_back(image_point_vgl);
			world_points.push_back(world_point);
		}

		break;
	}
}

void CourtView::clearAnnotations()
{
	windowPoints.clear();
	imagePoints.clear();
	world_points.clear();
}
