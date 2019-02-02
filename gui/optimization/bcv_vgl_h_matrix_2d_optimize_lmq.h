//
//  bcv_vgl_h_matrix_2d_optimize_lmq.h
//  PointLineHomography
//
//  Created by jimmy on 12/13/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineHomography__bcv_vgl_h_matrix_2d_optimize_lmq__
#define __PointLineHomography__bcv_vgl_h_matrix_2d_optimize_lmq__

#include <iostream>
#include <vgl/algo/vgl_h_matrix_2d_optimize_lmq.h>

class bcv_vgl_h_matrix_2d_optimize_lmq : public vgl_h_matrix_2d_optimize_lmq {
public:
    //: Constructor from initial homography to be optimized
    bcv_vgl_h_matrix_2d_optimize_lmq(vgl_h_matrix_2d<double> const& initial_h);
    
    //:compute from matched points and point on lines
    // point_on_lines1 are from images
    bool optimize_pl(std::vector<vgl_homg_point_2d<double> > const& points1,
                     std::vector<vgl_homg_point_2d<double> > const& points2,
                     std::vector<std::vector<vgl_homg_point_2d<double>> > const& point_on_lines1,
                     std::vector<vgl_homg_line_2d<double> > const& lines2,
                     vgl_h_matrix_2d<double>& H);
    
    
};

#endif /* defined(__PointLineHomography__bcv_vgl_h_matrix_2d_optimize_lmq__) */
