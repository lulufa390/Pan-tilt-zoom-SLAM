//
//  eigen_geometry_util.h
//  PointLineReloc
//
//  Created by jimmy on 2017-05-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__eigen_geometry_util__
#define __PointLineReloc__eigen_geometry_util__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

using std::vector;

class EigenGeometryUtil
{
public:
    
    static Eigen::Matrix3d vector2SkewSymmetricMatrix(const Eigen::Vector3d & v);
};

namespace EigenX {
    
    // pan tilt of a point
    // pp: principal point
    // pan_tilt_focal_length: unit degree and pixel
    // point: a point in the image
    // point_pan_tilt: pan and tilt of this point
    // assumption: the image is from a PTZ camera
    // O(1)
    void pointPanTilt(const Eigen::Vector2f& pp,
                      const Eigen::Vector3f& pan_tilt_focal_length,
                      const Eigen::Vector2f& point,
                      Eigen::Vector2f& point_pan_tilt);
    
    void pointPanTilt(const Eigen::Vector2d& pp,
                      const Eigen::Vector3d& pan_tilt_focal_length,
                      const Eigen::Vector2d& point,
                      Eigen::Vector2d& point_pan_tilt);
    
    
    // calculate pan, tilt and focal length from two (estimated) pan, tilt and zooms
    // panTilt: pan, tilt angles in degrees
    // image_location:
    // pp: principle point in the image, unit pixel
    // ptz: estimated pan, tilt and focal length, in degree and pixel
    bool ptzFromTwoPoints(const Eigen::Vector2f& pan_tilt1,
                          const Eigen::Vector2f& pan_tilt2,
                          const Eigen::Vector2f& point1,
                          const Eigen::Vector2f& point2,
                          const Eigen::Vector2f& pp,
                          Eigen::Vector3f& ptz);
    
    bool ptzFromTwoPoints(const Eigen::Vector2d& pan_tilt1,
                          const Eigen::Vector2d& pan_tilt2,
                          const Eigen::Vector2d& point1,
                          const Eigen::Vector2d& point2,
                          const Eigen::Vector2d& pp,
                          Eigen::Vector3d& ptz);
    
    
    
};  // namespace EigenX

#endif /* defined(__PointLineReloc__eigen_geometry_util__) */
