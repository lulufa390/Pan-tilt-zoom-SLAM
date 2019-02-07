//
//  cvimage_view.hpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-02.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef cvimage_view_hpp
#define cvimage_view_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "cvui.h"   

using cv::Mat;

class CVImageView {

private:
	std::string window_name;
	cv::Point imagePos;

	cv::Size windowSize;
	cv::Size imageSize;

	Mat image;
	Mat image_after_scale;

protected:
	std::vector<cv::Point> windowPoints;

public:
	Mat frame;

public:
	CVImageView(std::string name);
	virtual ~CVImageView();

	std::string getWindowName() const;
	void setWindowName(std::string name);

	cv::Size getWindowSize() const;
	void setWindowSize(cv::Size size);

	cv::Point getImagePosition() const;
	void setImagePosition(cv::Point p);

	// set/get image
	Mat getImage() const;
	void setImage(const Mat& img);

	// set/get image Size
	cv::Size getImageSize() const;
	void setImageSize(cv::Size size);

	// exchange between window point and image point
	cv::Point windowPointForImagePoint(const cv::Point& p) const;
	cv::Point imagePointForWindowPoint(const cv::Point& p) const;

	void drawFrame();
	void drawPoint(cv::Point p);

	// interface
	// subclasses must draw annotation (point, line)


	virtual void annotate() = 0;
};

#endif /* cvimage_view_hpp */
