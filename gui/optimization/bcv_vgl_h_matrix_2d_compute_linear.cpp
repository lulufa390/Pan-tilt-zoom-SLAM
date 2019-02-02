//
//  bcv_vgl_h_matrix_2d_compute_linear.cpp
//  PointLineHomography
//
//  Created by jimmy on 12/9/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "bcv_vgl_h_matrix_2d_compute_linear.h"
#include <vgl/algo/vgl_norm_trans_2d.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_det.h>


constexpr int TM_UNKNOWNS_COUNT = 9;
constexpr double DEGENERACY_THRESHOLD = 0.00001;  // FSM. see below.

bcv_vgl_h_matrix_2d_compute_linear::bcv_vgl_h_matrix_2d_compute_linear(bool allow_ideal_points)
    :allow_ideal_points_(allow_ideal_points)
{
}

// modify from
// vxl/core/vgl/algo/vgl_norm_trans_2d.hxx
bool bcv_vgl_h_matrix_2d_compute_linear::solve_linear_problem(int equ_count,
                          vector<vgl_homg_point_2d<double> > const& p1,
                          vector<vgl_homg_point_2d<double> > const& p2,
                          vector<vgl_homg_line_2d<double>> const& l1,
                          vector<vgl_homg_line_2d<double>> const& l2,
                          vgl_h_matrix_2d<double>& H)
{
    //transform the point sets and fill the design matrix
    vnl_matrix<double> D(equ_count, TM_UNKNOWNS_COUNT);
    int n = (int)p1.size();
    int row = 0;
    for (int i = 0; i < n; i++) {
        D(row, 0) =  p1[i].x() * p2[i].w();
        D(row, 1) =  p1[i].y() * p2[i].w();
        D(row, 2) =  p1[i].w() * p2[i].w();
        D(row, 3) = 0;
        D(row, 4) = 0;
        D(row, 5) = 0;
        D(row, 6) = -p1[i].x() * p2[i].x();
        D(row, 7) = -p1[i].y() * p2[i].x();
        D(row, 8) = -p1[i].w() * p2[i].x();
        ++row;
        
        D(row, 0) = 0;
        D(row, 1) = 0;
        D(row, 2) = 0;
        D(row, 3) =  p1[i].x() * p2[i].w();
        D(row, 4) =  p1[i].y() * p2[i].w();
        D(row, 5) =  p1[i].w() * p2[i].w();
        D(row, 6) = -p1[i].x() * p2[i].y();
        D(row, 7) = -p1[i].y() * p2[i].y();
        D(row, 8) = -p1[i].w() * p2[i].y();
        ++row;
        
        if (allow_ideal_points_) {
            D(row, 0) =  p1[i].x() * p2[i].y();
            D(row, 1) =  p1[i].y() * p2[i].y();
            D(row, 2) =  p1[i].w() * p2[i].y();
            D(row, 3) = -p1[i].x() * p2[i].x();
            D(row, 4) = -p1[i].y() * p2[i].x();
            D(row, 5) = -p1[i].w() * p2[i].x();
            D(row, 6) = 0;
            D(row, 7) = 0;
            D(row, 8) = 0;
            ++row;
        }
    }
    
    int m = (int)l1.size();
    for (int i = 0; i<m; i++) {
        double x = l1[i].a();
        double y = l1[i].b();
        double w = l1[i].c();
        double xx = l2[i].a();
        double yy = l2[i].b();
        double ww = l2[i].c();
        D(row, 0) = w*xx;
        D(row, 1) = 0;
        D(row, 2) = -x*xx;
        D(row, 3) = w*yy;
        D(row, 4) = 0;
        D(row, 5) = -x*yy;
        D(row, 6) = w*ww;
        D(row, 7) = 0;
        D(row, 8) = -x*ww;
        ++row;
        
        D(row, 0) = 0;
        D(row, 1) = w*xx;
        D(row, 2) = -y*xx;
        D(row, 3) = 0;
        D(row, 4) = w*yy;
        D(row, 5) = -y*yy;
        D(row, 6) = 0;
        D(row, 7) = w*ww;
        D(row, 8) = -y*ww;
        ++row;
        
        //@todo idea line
    }
    
    D.normalize_rows();
    vnl_svd<double> svd(D);
    
    //
    // FSM added :
    //
    if (svd.W(7)<DEGENERACY_THRESHOLD*svd.W(8)) {
        std::cerr << "vgl_h_matrix_2d_compute_linear : design matrix has rank < 8\n"
        << "vgl_h_matrix_2d_compute_linear : probably due to degenerate point configuration\n";
        return false;
    }
    // form the matrix from the nullvector
    H.set(svd.nullvector().data_block());
    return true;    
}

bool bcv_vgl_h_matrix_2d_compute_linear::compute_pl(vector<vgl_homg_point_2d<double> > const& points1,
                vector<vgl_homg_point_2d<double> > const& points2,
                vector<vgl_homg_line_2d<double> > const& lines1,
                vector<vgl_homg_line_2d<double> > const& lines2,
                vgl_h_matrix_2d<double>& H)
{
    // check input
    //number of points must be the same
    assert(points1.size() == points2.size());
    int np = (int)points1.size();
    //number of lines must be the same
    assert(lines1.size() == lines2.size());
    int nl = (int)lines1.size();
    
    int equ_count = np * (allow_ideal_points_ ? 3 : 2) + 2*nl;
    if ((np+nl)*2+1 < TM_UNKNOWNS_COUNT)
    {
        std::cerr << "vgl_h_matrix_2d_compute_linear: Need at least 4 matches.\n";
        if (np+nl == 0) std::cerr << "Could be std::vector setlength idiosyncrasies!\n";
        return false;
    }
    
    // normalize points
    vgl_norm_trans_2d<double> tr1, tr2;
    if (!tr1.compute_from_points(points1))
        return false;
    if (!tr2.compute_from_points(points2))
        return false;
    
    std::vector<vgl_homg_point_2d<double> > tpoints1, tpoints2;
    for (int i = 0; i<np; i++)
    {
        tpoints1.push_back(tr1(points1[i]));
        tpoints2.push_back(tr2(points2[i]));
    }
    
    // normalize lines
    // Combining Line and Point Correspondences for Homography Estimation
    // Equation 16
    vector<vgl_homg_line_2d<double>> tlines1;
    for (const auto& l: lines1) {
        double s  = tr1.get_matrix()[0][0];
        double tx = tr1.get_matrix()[0][2];
        double ty = tr1.get_matrix()[1][2];
        double a = l.a();
        double b = l.b();
        double c = l.c();
        double aa = s * a;
        double bb = s * b;
        double cc = s * (s * c - tx * a - ty * b);
        tlines1.push_back(vgl_homg_line_2d<double>(aa, bb, cc));
    }
    
    vector<vgl_homg_line_2d<double>> tlines2;
    for (const auto& l: lines2) {
        double s  = tr2.get_matrix()[0][0];
        double tx = tr2.get_matrix()[0][2];
        double ty = tr2.get_matrix()[1][2];
        double a = l.a();
        double b = l.b();
        double c = l.c();
        double aa = s * a;
        double bb = s * b;
        double cc = s * (s * c - tx * a - ty * b);
        tlines2.push_back(vgl_homg_line_2d<double>(aa, bb, cc));
    }
    
    assert(tlines1.size() == tlines2.size());
    
    vgl_h_matrix_2d<double> hh;
    if (!solve_linear_problem(equ_count, tpoints1, tpoints2, tlines1, tlines2, hh))
        return false;
    //
    // Next, hh has to be transformed back to the coordinate system of
    // the original point sets, i.e.,
    //  p1' = tr1 p1 , p2' = tr2 p2
    // hh was determined from the transform relation
    //  p2' = hh p1', thus
    // (tr2 p2) = hh (tr1 p1)
    //  p2 = (tr2^-1 hh tr1) p1 = H p1
    vgl_h_matrix_2d<double> tr2_inv = tr2.get_inverse();
    H = tr2_inv*hh*tr1;
    
    // cheirality
    if ( vnl_det(H.get_matrix()) < 0 )
    {
        H = vgl_h_matrix_2d<double>(-H.get_matrix());
    }
    return true;
}
