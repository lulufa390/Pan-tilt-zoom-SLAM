//
//  bcv_vgl_h_matrix_2d_compute_linear.h
//  PointLineHomography
//
//  Created by jimmy on 12/9/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineHomography__bcv_vgl_h_matrix_2d_compute_linear__
#define __PointLineHomography__bcv_vgl_h_matrix_2d_compute_linear__

#include <iostream>
#include <vector>
#include <vgl/vgl_homg_point_2d.h>
#include <vgl/vgl_homg_line_2d.h>
#include <vgl/algo/vgl_h_matrix_2d_compute.h>
#include <vgl/algo/vgl_h_matrix_2d_compute_linear.h>

using std::vector;

class bcv_vgl_h_matrix_2d_compute_linear: public vgl_h_matrix_2d_compute_linear {
    
private:
    bool allow_ideal_points_;
    
public:
    //:Assumes all corresponding points have equal weight
    bool solve_linear_problem(int equ_count,
                              vector<vgl_homg_point_2d<double> > const& p1,
                              vector<vgl_homg_point_2d<double> > const& p2,
                              vector<vgl_homg_line_2d<double>> const& l1,
                              vector<vgl_homg_line_2d<double>> const& l2,
                              vgl_h_matrix_2d<double>& H);
    
    // :compute from matched points and lines
    bool compute_pl(vector<vgl_homg_point_2d<double> > const& points1,
                    vector<vgl_homg_point_2d<double> > const& points2,
                    vector<vgl_homg_line_2d<double> > const& lines1,
                    vector<vgl_homg_line_2d<double> > const& lines2,
                    vgl_h_matrix_2d<double>& H) override;
public:
    bcv_vgl_h_matrix_2d_compute_linear(bool allow_ideal_points = false);
    
};



#endif /* defined(__PointLineHomography__bcv_vgl_h_matrix_2d_compute_linear__) */
