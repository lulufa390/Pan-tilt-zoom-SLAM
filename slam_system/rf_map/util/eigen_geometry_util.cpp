//
//  eigen_geometry_util.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-05-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "eigen_geometry_util.h"
#include <math.h>
#include <Eigen/Geometry>
#include <iostream>

using Eigen::Vector2f;
using Eigen::Vector3f;
using std::cout;
using std::endl;

Eigen::Matrix3d EigenGeometryUtil::vector2SkewSymmetricMatrix(const Eigen::Vector3d & v)
{
    // https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    double a = v.x();
    double b = v.y();
    double c = v.z();
    m << 0, -c, b,
    c, 0, -a,
    -b, a, 0;
    
    return m;
}


namespace EigenX {
    
    // @brief internal function. RPE report 2015 summer
    static bool focalLengthEstimation(const double a, const double b,
                                      const double c, const double d,
                                      double& fl,
                                      bool verbose = true)
    {
        double delta = (d*d *(a+b) - 2.0*c)*(d*d *(a+b) - 2.0*c) - 4.0 *(d*d*a*b-c*c)*(d*d-1.0);
        if (delta < 0.0) {
            if (verbose) {
                printf("Error: sqrt a negative number.\n");
                printf("delta is %f\n", delta);
            }
            return false;
        }
        
        double numerator   = 2.0 * (d*d*a*b - c*c);
        double denominator = 2.0*c - d*d *(a+b) + sqrt(delta);
        if (denominator == 0.0) {
            if (verbose) {
                printf("Error: denominator is zero.\n");
            }
            
            return false;
        }
        double fl_2 = numerator/denominator;
        if (fl_2 <= 0.0) {
            if (verbose) {
                printf("Error: focal length is not a real number. f^2 is %f\n", fl_2);
                printf("numerator, denominator is %f %f\n", numerator, denominator);
            }
            return false;
        }
        fl = sqrt(fl_2);
        return true;
    }
    
    void pointPanTilt(const Eigen::Vector2f& pp,
                      const Eigen::Vector3f& ptz,
                      const Eigen::Vector2f& point,
                      Eigen::Vector2f& point_pan_tilt)
    {
        double dx = point.x() - pp.x();
        double dy = point.y() - pp.y();
        double fl = ptz.z();
        double delta_pan = atan2(dx, fl) * 180.0/M_PI;
        double delta_tilt = atan2(dy, fl) * 180.0/M_PI;
        point_pan_tilt[0] = ptz[0] + delta_pan;
        point_pan_tilt[1] = ptz[1] - delta_tilt; // oppositive direction of y
    }
    
    void pointPanTilt(const Eigen::Vector2d& pp,
                      const Eigen::Vector3d& ptz,
                      const Eigen::Vector2d& point,
                      Eigen::Vector2d& point_pan_tilt)
    {
        double dx = point.x() - pp.x();
        double dy = point.y() - pp.y();
        double fl = ptz.z();
        double delta_pan = atan2(dx, fl) * 180.0/M_PI;
        double delta_tilt = atan2(dy, fl) * 180.0/M_PI;
        point_pan_tilt[0] = ptz[0] + delta_pan;
        point_pan_tilt[1] = ptz[1] - delta_tilt; // oppositive direction of y        
    }

    static Eigen::Matrix3f matrixFromPanYTiltX(double pan, double tilt)
    {
        Eigen::Matrix3f m;
        
        pan  *= M_PI / 180.0;
        tilt *= M_PI / 180.0;
        
        Eigen::Matrix3f R_tilt;
        R_tilt(0 ,0) = 1;   R_tilt(0, 1) = 0;          R_tilt(0, 2) = 0;
        R_tilt(1, 0) = 0;   R_tilt(1, 1) = cos(tilt);  R_tilt(1, 2) = sin(tilt);
        R_tilt(2, 0) = 0;   R_tilt(2, 1) = -sin(tilt);  R_tilt(2, 2) = cos(tilt);
        
        Eigen::Matrix3f R_pan;
        R_pan(0, 0) = cos(pan);   R_pan(0, 1) = 0;   R_pan(0, 2) = -sin(pan);
        R_pan(1, 0) = 0;          R_pan(1, 1) = 1;   R_pan(1, 2) = 0;
        R_pan(2, 0) = sin(pan);   R_pan(2, 1) = 0;   R_pan(2, 2) = cos(pan);
        
        m = R_tilt * R_pan;
        return m;
    }

    
    
    bool ptzFromTwoPoints(const Eigen::Vector2f& pan_tilt1,
                          const Eigen::Vector2f& pan_tilt2,
                          const Eigen::Vector2f& point1,
                          const Eigen::Vector2f& point2,
                          const Eigen::Vector2f& pp,
                          Eigen::Vector3f& ptz)
    {
        Eigen::Vector2f p1 = point1 - pp;
        Eigen::Vector2f p2 = point2 - pp;
       
        // Step 1. focal length
        double pan1  = pan_tilt1[0];
        double tilt1 = pan_tilt1[1];
        double pan2  = pan_tilt2[0];
        double tilt2 = pan_tilt2[1];
        
        double a = p1.dot(p1);
        double b = p2.dot(p2);
        double c = p1.dot(p2);
        Eigen::Vector3f z_axis = Eigen::Vector3f::Zero(3, 1);
        z_axis[2] = 1.0f;
        // rotate matrix of axis z
        Eigen::Vector3f rotated_axis_z = matrixFromPanYTiltX(pan2 - pan1, tilt2 - tilt1) * z_axis;
        double d = z_axis.dot(rotated_axis_z);  // angular difference of two angles
        double fl = 0;
        bool is_estimated = focalLengthEstimation(a, b, c, d, fl, false);
        if (!is_estimated) {
            return false;
        }
        
        // Step 2. pan and tilt
        double theta1 = pan1 - atan2(p1(0), fl)*180.0/M_PI;
        double theta2 = pan2 - atan2(p2(0), fl)*180.0/M_PI;
        ptz[0] = (theta1 + theta2)/2.0;
        
        double phi1 = tilt1 + atan2(p1(1), fl)*180.0/M_PI;  // oppositive direction as y is from top to down in image
        double phi2 = tilt2 + atan2(p2(1), fl)*180.0/M_PI;
        ptz[1] = (phi1 + phi2)/2.0;
        ptz[2] = fl;
        
        return true;
    }
    
    bool ptzFromTwoPoints(const Eigen::Vector2d& pan_tilt1,
                          const Eigen::Vector2d& pan_tilt2,
                          const Eigen::Vector2d& point1,
                          const Eigen::Vector2d& point2,
                          const Eigen::Vector2d& pp,
                          Eigen::Vector3d& ptz)
    {
        Eigen::Vector3f ptz_temp;
        bool is_ok = ptzFromTwoPoints(Eigen::Vector2f(pan_tilt1.x(), pan_tilt1.y()),
                                      Eigen::Vector2f(pan_tilt2.x(), pan_tilt2.y()),
                                      Eigen::Vector2f(point1.x(), point1.y()),
                                      Eigen::Vector2f(point2.x(), point2.y()),
                                      Eigen::Vector2f(pp.x(), pp.y()),
                                      ptz_temp);
        ptz[0] = ptz_temp[0];
        ptz[1] = ptz_temp[1];
        ptz[2] = ptz_temp[2];
        return is_ok;        
    }

    
}; // namespace EigenX