//
//  bcv_vgl_h_matrix_2d_optimize_lmq.cpp
//  PointLineHomography
//
//  Created by jimmy on 12/13/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "bcv_vgl_h_matrix_2d_optimize_lmq.h"

#include <algorithm>
#include <iterator>

#include <cassert>
#include <vnl/vnl_inverse.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>

#include <vgl/vgl_distance.h>


namespace  {
    class bcv_projection_lsqf : public vnl_least_squares_function
    {
        const std::vector<vgl_homg_point_2d<double> > from_points_;
        const std::vector<vgl_homg_point_2d<double> > to_points_;
        
        const std::vector<std::vector<vgl_homg_point_2d<double>> > from_point_on_lines_;
        const std::vector<vgl_homg_line_2d<double> > to_lines_;
        unsigned long n1_;
        unsigned long n2_;
        
        
    public:
        bcv_projection_lsqf(std::vector<vgl_homg_point_2d<double> > const& from_points,
                            std::vector<vgl_homg_point_2d<double> > const& to_points,
                            std::vector<std::vector<vgl_homg_point_2d<double>> > const& from_point_on_lines,
                            std::vector<vgl_homg_line_2d<double> > const& to_lines,
                            const int constraint_num)
        : vnl_least_squares_function(9, constraint_num, no_gradient),
        from_points_(from_points),
        to_points_(to_points),
        from_point_on_lines_(from_point_on_lines),
        to_lines_(to_lines)
        {
            n1_ = from_points_.size();
            n2_ = from_point_on_lines_.size();
            
            assert(n1_ == to_points.size());
            assert(n2_ == to_lines.size());
            assert(constraint_num >= 9);
        }
        
        ~bcv_projection_lsqf() override = default;
        
        //: compute the projection error given a set of h parameters.
        // The residuals required by f are the Euclidean x and y coordinate
        // differences between the projected from points and the
        // corresponding to points.
        // and the project points to the line
        void f(const vnl_vector<double>& hv, vnl_vector<double>& proj_err) override
        {
            assert(hv.size()==9);
            // project and compute residual
            vgl_h_matrix_2d<double> h(hv.data_block());
            unsigned long k = 0;
            for (unsigned i = 0; i<n1_; ++i)
            {
                vgl_homg_point_2d<double> to_proj = h(from_points_[i]);
                vgl_point_2d<double> p_proj(to_proj);
                double xp = to_points_[i].x(), yp = to_points_[i].y();
                double xproj = p_proj.x(), yproj = p_proj.y();
                double dx = xp - xproj;
                double dy = yp - yproj;
                proj_err[k] = dx;
                k++;
                proj_err[k] = dy;
                k++;
            }
            for (unsigned i = 0; i<n2_; ++i) {
                const vgl_homg_line_2d<double> line = to_lines_[i];
                for (const vgl_homg_point_2d<double>& p: from_point_on_lines_[i]) {
                    vgl_homg_point_2d<double> to_proj = h(p);
                    double dist = vgl_distance(to_proj, line);
                    proj_err[k] = dist;
                    k++;
                }
            }
            proj_err[k]=1.0-hv.magnitude();
            k++;
        }
        
        void reprojection_error(const vnl_vector<double>& hv) {
            assert(hv.size()==9);
            
            // project and compute residual
            std::vector<double> point_errors;
            vgl_h_matrix_2d<double> h(hv.data_block());
            for (unsigned i = 0; i<n1_; ++i)
            {
                vgl_homg_point_2d<double> to_proj = h(from_points_[i]);
                vgl_point_2d<double> p_proj(to_proj);
                double xp = to_points_[i].x(), yp = to_points_[i].y();
                double xproj = p_proj.x(), yproj = p_proj.y();
                double dx = xp - xproj;
                double dy = yp - yproj;
                point_errors.push_back(sqrt(dx*dx + dy*dy));
            }
            std::vector<double> line_errors;
            for (unsigned i = 0; i<n2_; ++i) {
                const vgl_homg_line_2d<double> line = to_lines_[i];
                for (const vgl_homg_point_2d<double>& p: from_point_on_lines_[i]) {
                    vgl_homg_point_2d<double> to_proj = h(p);
                    double dist = vgl_distance(to_proj, line);
                    line_errors.push_back(dist);
                }
            }
            std::cout<<"point to point reprojection error: "<<std::endl;
            std::copy(point_errors.begin(), point_errors.end(), std::ostream_iterator<double>(std::cout, " "));
            std::cout<<std::endl;
            std::cout<<"point to line distance: "<<std::endl;
            std::copy(line_errors.begin(), line_errors.end(), std::ostream_iterator<double>(std::cout, " "));
            std::cout<<std::endl;
        }
    };
}

bcv_vgl_h_matrix_2d_optimize_lmq::
bcv_vgl_h_matrix_2d_optimize_lmq(vgl_h_matrix_2d<double> const& initial_h)
: vgl_h_matrix_2d_optimize_lmq(initial_h)
{
}

bool bcv_vgl_h_matrix_2d_optimize_lmq::
optimize_pl(std::vector<vgl_homg_point_2d<double> > const& points1,
                 std::vector<vgl_homg_point_2d<double> > const& points2,
                 std::vector<std::vector<vgl_homg_point_2d<double>> > const& point_on_lines1,
                 std::vector<vgl_homg_line_2d<double> > const& lines2,
                 vgl_h_matrix_2d<double>& h_optimized)
{
    assert(points1.size() == points2.size());
    
    int constraint_num = (int)points1.size() * 2;
    for (const auto& item: point_on_lines1) {
        constraint_num += (int)item.size();
    }
    constraint_num += 1;
    
    bcv_projection_lsqf lsq(points1, points2, point_on_lines1, lines2, constraint_num);
    vnl_vector<double> hv(9, 0);
    vnl_matrix_fixed<double,3,3> m =  initial_h_.get_matrix();
    unsigned i = 0;
    for (unsigned r=0; r<3; ++r) {
        for (unsigned c=0; c<3; ++c, ++i) {
            hv[i] = m[r][c];
        }
    }
    vnl_levenberg_marquardt lm(lsq);
    lm.set_verbose(verbose_);
    lm.set_trace(trace_);
    lm.set_x_tolerance(htol_);
    lm.set_f_tolerance(ftol_);
    lm.set_g_tolerance(gtol_);
    bool success = lm.minimize(hv);
    if (verbose_)
    {
        lm.diagnose_outcome(std::cout);
    }    
    if (success)
        h_optimized.set(hv.data_block());
    else
        h_optimized = initial_h_;
    //lsq.reprojection_error(hv);
    if ( vnl_det(h_optimized.get_matrix()) < 0 )
    {
        h_optimized = vgl_h_matrix_2d<double>(-h_optimized.get_matrix());
    }
    return success;
}
