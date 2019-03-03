//
//  feature_annotation_view.hpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-02.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef feature_annotation_view_hpp
#define feature_annotation_view_hpp

#include <stdio.h>
#include <vgl/vgl_point_2d.h>
#include <vector>
#include "cvimage_view.hpp"

using std::vector;

// a view shows the image and use can annotate features (point, line)
class FeatureAnnotationView : public CVImageView {

private:
	vector<cv::Point> image_points_;

	vector<vgl_line_segment_2d<double>> lines_;

	vector<vgl_point_2d<double>> circle_points_;

public:
	FeatureAnnotationView(std::string name);
	~FeatureAnnotationView();

	// get points on image coordinate
	vector<vgl_point_2d<double>> getPoints() const;

	// get line segment
	vector<vgl_line_segment_2d<double>> getLines() const;

	// get image point that are on circles
	vector<vgl_point_2d<double>> getCirclePoints() const;

	virtual void annotate();
	virtual void clearAnnotations();
};

#endif /* feature_annotation_view_hpp */
