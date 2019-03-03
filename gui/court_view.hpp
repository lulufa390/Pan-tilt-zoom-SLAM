//
//  court_view.hpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-02.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef court_view_hpp
#define court_view_hpp

#include <stdio.h>
#include <memory>
#include <vgl/vgl_point_2d.h>
#include <vector>
#include "cvimage_view.hpp"
#include "court/play_field.h"

using std::vector;
class CourtView : public CVImageView {

private:
	std::shared_ptr<PlayField> play_field_;

	//std::vector<vgl_point_2d<double>> image_points_;
	std::vector<vgl_point_2d<double>> world_points_;


public:
	CourtView(std::string name);
	~CourtView();

	// get points on world coordinate, unit meter
	vector<vgl_point_2d<double>> getPoints() const;
    
    

	virtual void annotate();
	virtual void clearAnnotations();
};

#endif /* court_view_hpp */
