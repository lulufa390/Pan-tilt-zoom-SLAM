//
//  cvimage_view.cpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-02.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "cvimage_view.hpp"

CVImageView::CVImageView(std::string name)
{
	// default parameters for CVImageView
	windowSize = cv::Size(1300, 740);
	imageSize = cv::Size(1280, 720);
	imagePos = cv::Point(10, 10);

	window_name = name;

	frame = cv::Mat(windowSize.height, windowSize.width, CV_8UC3);
	frame = cv::Scalar(50, 50, 50);

	imageAfterScale = cv::Mat(imageSize.height, imageSize.width, CV_8UC3, cv::Scalar(0, 0, 0));


}

CVImageView::~CVImageView() {}

cv::Size CVImageView::getWindowSize() const
{
	return windowSize;
}

void CVImageView::setWindowSize(cv::Size size)
{
	windowSize = size;
}

std::string CVImageView::getWindowName() const
{
	return window_name;
}

void CVImageView::setWindowName(std::string name)
{
	window_name = name;
}

// set/get image
Mat CVImageView::getImage() const
{
	return image;
}

void CVImageView::setImage(const Mat& img)
{
	image = img;
	cv::resize(image, imageAfterScale, imageSize, cv::INTER_LINEAR);
}

cv::Point CVImageView::getImagePosition() const
{
	return imagePos;
}

void CVImageView::setImagePosition(cv::Point p)
{
	imagePos = p;
}

cv::Size CVImageView::getImageSize() const
{
	return imageSize;
}

void CVImageView::setImageSize(cv::Size size)
{
	imageSize = size;
	cv::resize(image, imageAfterScale, imageSize, cv::INTER_LINEAR);
}

// exchange between window point and image point
cv::Point CVImageView::windowPointForImagePoint(const cv::Point& p) const
{
	return cv::Point(0, 0);
}

cv::Point CVImageView::imagePointForWindowPoint(const cv::Point& p) const
{
	cv::Size beforeScale = image.size();
	cv::Size afterScale = imageAfterScale.size();

	double scale_x = 1.0 * (beforeScale.width - 1) / (afterScale.width - 1);
	double scale_y = 1.0 * (beforeScale.height - 1) / (afterScale.height - 1);

	int x = round((p.x - imagePos.x) * scale_x);
	int y = round((p.y - imagePos.y) * scale_y);

	return cv::Point(x, y);
}


void CVImageView::drawPoint(cv::Point p)
{
	cv::Point leftup = cv::Point(p.x - 5, p.y - 5);
	cv::Point leftdown = cv::Point(p.x - 5, p.y + 5);
	cv::Point rightup = cv::Point(p.x + 5, p.y - 5);
	cv::Point rightdown = cv::Point(p.x + 5, p.y + 5);
	cv::line(frame, leftup, rightdown, cv::Scalar(255, 0, 0), 1);
	cv::line(frame, leftdown, rightup, cv::Scalar(255, 0, 0), 1);
}

void CVImageView::drawFrame()
{
	cvui::image(frame, imagePos.x, imagePos.y, imageAfterScale);

	for (auto iter = windowPoints.begin(); iter != windowPoints.end(); iter++)
	{
		drawPoint(*iter);
	}
}