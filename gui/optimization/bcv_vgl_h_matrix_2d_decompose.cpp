//
//  bcv_vgl_h_matrix_2d_decompose.cpp
//  PointLineHomography
//
//  Created by jimmy on 12/22/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include <iostream>

#include "bcv_vgl_h_matrix_2d_decompose.h"
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_cross.h>


using std::cout;
using std::endl;

static double opposite_of_mirror(const vnl_matrix<double>&m,
                                 int row, int col) {
    assert(m.rows() == 3 && m.cols() == 3);
    // page 14
    int x1 = (col == 0)? 1: 0;
    int x2 = (col == 2)? 1: 2;
    int y1 = (row == 0)? 1: 0;
    int y2 = (row == 2)? 1: 2;
    return -(m[y1][x1] * m[y2][x2] - m[y1][x2] * m[y2][x1]);
}

static int sgn(const double val) {
    return val >= 0 ? 1:-1;
}

bool bcv_vgl_h_matrix_2d_decompose::compute(const vnl_matrix_fixed<double, 3, 3>& camera_matrix,
                                            const vnl_matrix_fixed<double, 3, 3>& homo,
                                            std::vector<vnl_matrix_fixed<double, 3, 3>>& rotations,
                                            std::vector<vnl_vector_fixed<double, 3>>& translations,
                                            std::vector<vnl_vector_fixed<double, 3>>& normals)
{
    const double epsilon = 0.001;
    // From "deeper understanding of the homography decomposition for vision-based control"
    vnl_matrix<double> K = camera_matrix.as_matrix();
    vnl_matrix<double> H = homo.as_matrix();
    vnl_matrix<double> invK = vnl_matrix_inverse<double>(K);
    
    // 1. normalize K
    H = invK * H * K;
    
    // 2. rescale H
    vnl_svd<double> svd(H);
    double scale = svd.W()[1];
    assert(scale != 0.0);
    H /= scale;
    
    // 3. estimate translation and rotation from H, translation is up to a scale   
    vnl_matrix<double> I = vnl_matrix<double>(3, 3);
    I.set_identity();
    
    vnl_matrix<double> S = H.transpose() * H - I;
    // homograph is a pure rotation matrix
    if (S.operator_inf_norm() < epsilon) {
        vnl_vector_fixed<double, 3> zero;
        zero.fill(0);
        rotations.push_back(vnl_matrix_fixed<double, 3, 3>(H));
        translations.push_back(zero);
        normals.push_back(zero);
        return true;
    }
    
    double s11 = S[0][0];
    double s12 = S[0][1];
    double s13 = S[0][2];
    double s22 = S[1][1];
    double s23 = S[1][2];
    double s33 = S[2][2];
    
    double Ms11 = opposite_of_mirror(S, 0, 0);
    double Ms22 = opposite_of_mirror(S, 1, 1);
    double Ms33 = opposite_of_mirror(S, 2, 2);
    
    assert(Ms11 >= 0);
    assert(Ms22 >= 0);
    assert(Ms33 >= 0);
    
    double rtMs11 = sqrt(Ms11);
    double rtMs22 = sqrt(Ms22);
    double rtMs33 = sqrt(Ms33);
    
    double Ms12 = opposite_of_mirror(S, 0, 1);
    double Ms23 = opposite_of_mirror(S, 1, 2);
    double Ms13 = opposite_of_mirror(S, 0, 2);
    
    double s11_abs = std::abs(s11);
    double s22_abs = std::abs(s22);
    double s33_abs = std::abs(s33);
    
    // page 14
    // compute na and nb using the sii with largest absolute value
    int max_index = 0;
    if (s11_abs < s22_abs){
        max_index = 1;
        if (s22_abs < s33_abs) {
            max_index = 2;
        }
    }
    else {
        if (s11_abs < s33_abs) {
            max_index = 2;
        }
    }
    
    vnl_vector<double> na = vnl_vector<double>(3);
    vnl_vector<double> nb = vnl_vector<double>(3);
    
    // equation 11 - 13
    int e23 = sgn(Ms23);
    int e13 = sgn(Ms13);
    int e12 = sgn(Ms12);
    switch (max_index) {
        case 0:
            na[0] = s11;
            na[1] = s12 + rtMs33;
            na[2] = s13 + e23 * rtMs22;
            
            nb[0] = s11;
            nb[1] = s12 - rtMs33;
            nb[2] = s13 - e23 * rtMs22;
            break;
            
        case 1:
            na[0] = s12 + rtMs33;
            na[1] = s22;
            na[2] = s23 - e13 * rtMs11;
            
            nb[0] = s12 - rtMs33;
            nb[1] = s22;
            nb[2] = s23 + e13 * rtMs11;
            break;
            
        case 2:
            na[0] = s13 + e12 * rtMs22;
            na[1] = s23 + rtMs11;
            na[2] = s33;
            
            nb[0] = s13 - e12 * rtMs22;
            nb[1] = s23 - rtMs11;
            nb[2] = s33;
            break;
            
        default:
            assert(0);
            break;
    }
    
    // normalize na nb
    na = na.normalize();
    nb = nb.normalize();
    
    // compute translation
    double trace_S = S[0][0] + S[1][1] + S[2][2];
    double v = 2 * sqrt(1 + trace_S - Ms11 - Ms22 - Ms33);
    
    double t_e2 = 2 + trace_S - v;
    double rho2 = 2 + trace_S + v;
    assert(t_e2 >= 0);
    assert(rho2 >= 0);
    
    double t_e = sqrt(t_e2);
    double rho = sqrt(rho2);
    
    int sign_sii = sgn(S[max_index][max_index]);
    // quation 16, 17
    vnl_vector<double> ta_star = 0.5 * t_e * (sign_sii * rho * nb - t_e * na);
    vnl_vector<double> tb_star = 0.5 * t_e * (sign_sii * rho * na - t_e * nb);
    
    
    vnl_matrix<double> Ra = H * (I - 2.0/v * outer_product(ta_star, na));
    vnl_matrix<double> Rb = H * (I - 2.0/v * outer_product(tb_star, nb));
    assert(Ra.rows() == 3 && Ra.cols() == 3);
    assert(Rb.rows() == 3 && Rb.cols() == 3);
    
    //cout<<Ra<<endl;
    vnl_vector<double> ta = Ra * ta_star;
    vnl_vector<double> tb = Rb * tb_star;
    
    rotations.resize(4);
    translations.resize(4);
    normals.resize(4);
    // four possible solutions
    // Ra ta na
    // Ra -ta -na
    // Rb tb  nb
    // Rb -tb -nb
    rotations[0] = vnl_matrix_fixed<double, 3, 3>(Ra);
    translations[0] = vnl_vector_fixed<double, 3>(ta);
    normals[0] = vnl_vector_fixed<double, 3>(na);
    
    
    rotations[1] = vnl_matrix_fixed<double, 3, 3>(Ra);
    translations[1] = vnl_vector_fixed<double, 3>(-ta);
    normals[1] = vnl_vector_fixed<double, 3>(-na);
    
    rotations[2] = vnl_matrix_fixed<double, 3, 3>(Rb);
    translations[2] = vnl_vector_fixed<double, 3>(tb);
    normals[2] = vnl_vector_fixed<double, 3>(nb);
    
    rotations[3] = vnl_matrix_fixed<double, 3, 3>(Rb);
    translations[3] = vnl_vector_fixed<double, 3>(-tb);
    normals[3] = vnl_vector_fixed<double, 3>(-nb);
    
    return true;
}

bool bcv_vgl_h_matrix_2d_decompose::compute(const vnl_matrix_fixed<double, 3, 3>& camera_matrix,
                                            const vnl_matrix_fixed<double, 3, 3>& homo,
                                            std::vector<vnl_matrix_fixed<double, 3, 3>>& rotations,
                                            std::vector<vnl_vector_fixed<double, 3>>& translations)
{
    vnl_matrix<double> K = camera_matrix.as_matrix();
    vnl_matrix<double> H = homo.as_matrix();
    vnl_matrix<double> invK = vnl_matrix_inverse<double>(K);
    
    // H from world coordinate to camera coordinate
    H = invK * H;
    
    vnl_vector<double> hc0, hc1, hc2;
    hc0 = H.get_column(0);
    H /= hc0.two_norm();
    
    // first solution
    {
        vnl_vector<double> t = H.get_column(2);
        hc0 = H.get_column(0);
        hc1 = H.get_column(1);
        hc2 = vnl_cross_3d(hc0, hc1);
        
        vnl_matrix_fixed<double, 3, 3> r;
        r.set_column(0, hc0);
        r.set_column(1, hc1);
        r.set_column(2, hc2);
        
        rotations.push_back(r);
        translations.push_back(t);
    }
    
    H *= -1.0;
    // second solution
    {
        vnl_vector<double> t = H.get_column(2);
        hc0 = H.get_column(0);
        hc1 = H.get_column(1);
        hc2 = vnl_cross_3d(hc0, hc1);
        
        vnl_matrix_fixed<double, 3, 3> r;
        r.set_column(0, hc0);
        r.set_column(1, hc1);
        r.set_column(2, hc2);
        
        rotations.push_back(r);
        translations.push_back(t);
    }
    
    return true;
}


