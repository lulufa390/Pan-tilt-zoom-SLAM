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
#include <opencv2/core/core.hpp>

using cv::Mat;
class CVImageView {
private:
    
public:
    CVImageView();
    virtual ~CVImageView();
    
    void setWindowSize(int width, int height);
    
    // set/get image
    Mat getImage()const;
    void setImage(const Mat& image);
    
    double imageScale()const;
    void setImageScale();
    
    // exchange between window point and image point
    cv::Point windowPointForImagePoint(const cv::Point& p) const;
    cv::Point imagePointForWindowPoint(const cv::Point& p) const;
    
    // interface
    // subclasses must draw annotation (point, line)
    virtual void annotate() = 0;
};

#endif /* cvimage_view_hpp */
