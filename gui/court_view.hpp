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
#include <vgl/vgl_point_2d.h>
#include <vector>
#include "cvimage_view.hpp"

using std::vector;
class CourtView: public CVImageView {
public:
    CourtView();
    ~CourtView();
    
    // get points on world coordinate, unit meter
    vector<vgl_point_2d<double>> getPoints() const;
    
    virtual void annotate();
    
};

#endif /* court_view_hpp */
