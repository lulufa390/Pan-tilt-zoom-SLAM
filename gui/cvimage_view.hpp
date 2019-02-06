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

using cv::Mat;

class CVImageView {
private:
	std::string window_name;
	cv::Point imagePos;
	
	cv::Size windowSize;
	cv::Size imageSize;

	Mat frame;
	Mat image;

	std::vector<cv::Point> windowPoints;

public:
    CVImageView();
    virtual ~CVImageView();
    
	Mat getFrame() const;
	std::string getWindowName();

	cv::Size getWindowSize() const;
    void setWindowSize(cv::Size size);
	
	cv::Point getImagePosition() const;
	void setImagePosition(cv::Point p);
    
    // set/get image
    Mat getImage() const;
    void setImage(const Mat& image);
    
	// set/get image Size
    cv::Size getImageSize() const;
    void setImageSize(cv::Size size);

	
    // exchange between window point and image point
    cv::Point windowPointForImagePoint(const cv::Point& p) const;
    cv::Point imagePointForWindowPoint(const cv::Point& p) const;
    

	void draw_frame();

    // interface
    // subclasses must draw annotation (point, line)
    virtual void annotate() = 0;
};

#endif /* cvimage_view_hpp */
