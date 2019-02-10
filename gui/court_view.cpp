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
        
        vgl_point_2d<double> p = vgl_point_2d<double>(imagePoint.x, imagePoint.y);
        vgl_point_2d<double> q;
        
        bool isFind = play_field_->find_candinate_point(p, q, 10);
        if (isFind) {
           // points.push_back(q);
           // NSLog(@"find position %f %f\n", q.x(), q.y());
            printf("find court points. ");
        }
            

		windowPoints.push_back(windowPoint);
		imagePoints.push_back(imagePoint);
		break;
	}
}

void CourtView::clearAnnotations()
{
	windowPoints.clear();
	imagePoints.clear();
}
