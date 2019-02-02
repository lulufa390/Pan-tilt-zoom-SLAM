//
//  bcv_vgl_h_matrix_2d_decompose.h
//  PointLineHomography
//
//  Created by jimmy on 12/22/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineHomography__bcv_vgl_h_matrix_2d_decompose__
#define __PointLineHomography__bcv_vgl_h_matrix_2d_decompose__

#include <vgl/algo/vgl_h_matrix_2d.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vector>


class bcv_vgl_h_matrix_2d_decompose {
private:
    bool positive_z_;
public:
    bcv_vgl_h_matrix_2d_decompose(bool positive_z = true) {
        positive_z_ = positive_z;
    }
    
    ~bcv_vgl_h_matrix_2d_decompose() = default;
    
    // decompose camera pose from camera matrix (K) and homogrpahy matrix
    // output: 4 solutions, translation is up to a scale factor
    bool compute(const vnl_matrix_fixed<double, 3, 3>& camera_matrix,
                 const vnl_matrix_fixed<double, 3, 3>& homo,
                 std::vector<vnl_matrix_fixed<double, 3, 3>>& rotations,
                 std::vector<vnl_vector_fixed<double, 3>>& translations,
                 std::vector<vnl_vector_fixed<double, 3>>& normals);
    
    // assume homogrpahy is world coordinate to image coordinate
    // output: 2 solutions
    bool compute(const vnl_matrix_fixed<double, 3, 3>& camera_matrix,
                 const vnl_matrix_fixed<double, 3, 3>& homo,
                 std::vector<vnl_matrix_fixed<double, 3, 3>>& rotations,
                 std::vector<vnl_vector_fixed<double, 3>>& translations);
  
};

#endif /* defined(__PointLineHomography__bcv_vgl_h_matrix_2d_decompose__) */
