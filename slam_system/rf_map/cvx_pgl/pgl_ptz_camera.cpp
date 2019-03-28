//
//  pgl_ptz_camera.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "pgl_ptz_camera.h"
#include <iostream>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


using cvx_gl::rotation_3d;
using cvx_pgl::calibration_matrix;
using std::cout;
using std::endl;


namespace cvx_pgl  {
    bool ptz_camera::set_camera(const perspective_camera& camera,
                                const MatrixXd & wld_pts,
                                const MatrixXd & img_pts)
    {
        assert(wld_pts.rows() == img_pts.rows());
        assert(wld_pts.rows() >= 2);
        K_ = camera.get_calibration();
        camera_center_ = camera.get_camera_center();
        R_ = camera.get_rotation();
        recompute_matrix();
        
        perspective_camera estimatedCamera;
        bool isEsimated = ptz_camera::estimatePTZWithFixedBasePositionRotation(wld_pts, img_pts, camera,
                                                                               camera_center_, base_rotation_, ptz_, estimatedCamera);
        return isEsimated;
    }
    
    bool ptz_camera::set_camera(const perspective_camera& camera)
    {
        Eigen::Vector3d dif_cc = cc_ - camera.get_camera_center();
        if (dif_cc.norm() >= 1e-4) {
            printf("Warning: camera center is different: %f\n", dif_cc.norm());
            return false;
        }
        
        // camera base rotation
        Eigen::Matrix3d baseRInv = rotation_3d(base_rotation_).as_matrix().inverse();
        
        Eigen::Matrix3d R_pan_tilt = camera.get_rotation().as_matrix() * baseRInv;
        double cos_pan = R_pan_tilt(0, 0);
        double sin_pan = -R_pan_tilt(0, 2);
        double cos_tilt = R_pan_tilt(1, 1);
        double sin_tilt = -R_pan_tilt(2, 1);
        double pan  = atan2(sin_pan, cos_pan) * 180.0 /M_PI;
        double tilt = atan2(sin_tilt, cos_tilt) * 180.0 /M_PI;
        ptz_[0] = pan;
        ptz_[1] = tilt;
        ptz_[2] = camera.get_calibration().focal_length();
        
        K_ = camera.get_calibration();
        camera_center_ = camera.get_camera_center();
        R_ = camera.get_rotation();
        recompute_matrix();
        return true;
    }
    
    bool ptz_camera::set_ptz(const Vector3d& ptz)
    {
        ptz_ = ptz;
        
        Eigen::Matrix3d R = matrixFromPanYTiltX(pan(), tilt()) * cvx_gl::rotation_3d(base_rotation_).as_matrix();
        K_ = cvx_pgl::calibration_matrix(focal_length(), pp_);
        R_ = cvx_gl::rotation_3d(R);
        recompute_matrix();
        return true;
    }
    
    
    Eigen::Vector2d ptz_camera::project(double pan, double tilt) const
    {
        return panTilt2Point(pp_, ptz_, Eigen::Vector2d(pan, tilt));
    }
    
    Eigen::Vector2d ptz_camera::back_project(double x, double y) const
    {
        return point2PanTilt(pp_, ptz_, Eigen::Vector2d(x, y));
    }
    
    namespace {
        struct PTZFromPointFunctor
        {
            typedef double Scalar;
            
            typedef Eigen::VectorXd InputType;
            typedef Eigen::VectorXd ValueType;
            typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;
            
            enum {
                InputsAtCompileTime = Eigen::Dynamic,
                ValuesAtCompileTime = Eigen::Dynamic
            };
            
            const Eigen::MatrixXd wld_pts_;
            const Eigen::MatrixXd img_pts_;
            const Eigen::Vector2d principal_point_;
            const Eigen::Vector3d camera_center_;
            const Eigen::Matrix3d base_rotation_; //rodrigue angle of camera base rotation
            
            int m_inputs;
            int m_values;
            
            PTZFromPointFunctor(const Eigen::MatrixXd& wld_pts,
                                const Eigen::MatrixXd& img_pts,
                                const Eigen::Vector2d& pp,
                                const Eigen::Vector3d& cc,
                                const Eigen::Matrix3d& base_rotation):
            wld_pts_(wld_pts), img_pts_(img_pts), principal_point_(pp),
            camera_center_(cc), base_rotation_(base_rotation)
            {
                m_inputs = 3;
                m_values = (int)wld_pts_.rows() * 2;
                assert(m_values >= 4);
            }
            
            
            int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
            {
                double pan  = x[0];
                double tilt = x[1];
                double fl = x[2];
                
                calibration_matrix K(fl, principal_point_);
                Eigen::Matrix3d R_pan_tilt =  matrixFromPanYTiltX(pan, tilt);
                Eigen::Matrix3d R = R_pan_tilt * base_rotation_;
                
                perspective_camera camera;
                camera.set_calibration(K);
                camera.set_camera_center(camera_center_);
                camera.set_rotation(R);
                
                // loop each points
                for (int i = 0, idx = 0; i<wld_pts_.rows(); i++) {
                    Vector3d p = wld_pts_.row(i);
                    double u = 0.0, v = 0.0;
                    camera.project(p.x(), p.y(), p.z(), u, v);
                    
                    fx[idx++] = img_pts_.row(i).x() - u;
                    fx[idx++] = img_pts_.row(i).y() - v;
                }                
                return 0;
            }
            
            int inputs() const { return m_inputs; }// inputs is the dimension of x.
            int values() const { return m_values; } // "values" is the number of f_i and
            
            void getResult(const Eigen::VectorXd& x,
                           perspective_camera& camera)
            {
                double pan  = x[0];
                double tilt = x[1];
                double fl = x[2];
                
                calibration_matrix K(fl, principal_point_);
                Eigen::Matrix3d R_pan_tilt =  matrixFromPanYTiltX(pan, tilt);
                Eigen::Matrix3d R = R_pan_tilt * base_rotation_;
                
                camera.set_calibration(K);
                camera.set_camera_center(camera_center_);
                camera.set_rotation(R);
            }
            
            void reprojectionError(const Eigen::VectorXd& x,
                                   Eigen::MatrixXd & mean,
                                   Eigen::MatrixXd & cov)
            {
                double pan  = x[0];
                double tilt = x[1];
                double fl = x[2];
                
                calibration_matrix K(fl, principal_point_);
                Eigen::Matrix3d R_pan_tilt =  matrixFromPanYTiltX(pan, tilt);
                Eigen::Matrix3d R = R_pan_tilt * base_rotation_;
                
                perspective_camera camera;
                camera.set_calibration(K);
                camera.set_camera_center(camera_center_);
                camera.set_rotation(R);
                
                // loop each points
                Eigen::MatrixXd error(wld_pts_.rows(), 2);
                for (int i = 0; i<wld_pts_.rows(); i++) {
                    Vector3d p = wld_pts_.row(i);
                    double u = 0.0, v = 0.0;
                    camera.project(p.x(), p.y(), p.z(), u, v);
                    
                    error(i, 0) = img_pts_.row(i).x() - u;
                    error(i, 1) = img_pts_.row(i).y() - v;
                }
                mean = error.colwise().mean();
                MatrixXd centered = error.rowwise() - error.colwise().mean();
                cov = (centered.adjoint() * centered) / error.rows();
            }
        };
    }

    
    bool ptz_camera::estimatePTZWithFixedBasePositionRotation (const MatrixXd & wld_pts,
                                                                const MatrixXd & img_pts,
                                                                const perspective_camera & init_camera,
                                                                const Vector3d & camera_center,
                                                                const Vector3d & rod,
                                                                Vector3d & ptz,
                                                                perspective_camera & estimated_camera)
    {
        assert(wld_pts.rows() == img_pts.rows());
        assert(wld_pts.cols() == 3);
        assert(img_pts.cols() == 2);
        assert(wld_pts.rows() >= 2);
        
        // camera base rotation
        Eigen::Matrix3d baseRinv = rotation_3d(rod).as_matrix().inverse();
        
        // init values
        Eigen::VectorXd x(3);
        Eigen::Matrix3d R_pan_tilt = init_camera.get_rotation().as_matrix() * baseRinv;
        double cos_pan = R_pan_tilt(0, 0);
        double sin_pan = -R_pan_tilt(0, 2);
        double cos_tilt = R_pan_tilt(1, 1);
        double sin_tilt = -R_pan_tilt(2, 1);
        double pan  = atan2(sin_pan, cos_pan) * 180.0 /M_PI;
        double tilt = atan2(sin_tilt, cos_tilt) * 180.0 /M_PI;
        x[0] = pan;
        x[1] = tilt;
        x[2] = init_camera.get_calibration().focal_length();
        
        
        // optimize pan, tilt and focal length
        PTZFromPointFunctor opt_functor(wld_pts, img_pts, init_camera.get_calibration().principal_point(),
                                        init_camera.get_camera_center(), rotation_3d(rod).as_matrix());
        Eigen::NumericalDiff<PTZFromPointFunctor> numerical_dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PTZFromPointFunctor>, double> lm(numerical_dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = 100;
        
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
        printf("LevenbergMarquardt status %d\n", status);
        
        opt_functor.getResult(x, estimated_camera);
        ptz[0] = x[0];
        ptz[1] = x[1];
        ptz[2] = x[2];
        
        Eigen::MatrixXd mean;
        Eigen::MatrixXd cov;
        opt_functor.reprojectionError(x, mean, cov);
        return true;
    }
    
    Eigen::Matrix3d matrixFromPanYTiltX(double pan, double tilt)
    {
        Eigen::Matrix3d m;
        
        pan  *= M_PI / 180.0;
        tilt *= M_PI / 180.0;
        
        Eigen::Matrix3d R_tilt;
        R_tilt(0 ,0) = 1;   R_tilt(0, 1) = 0;          R_tilt(0, 2) = 0;
        R_tilt(1, 0) = 0;   R_tilt(1, 1) = cos(tilt);  R_tilt(1, 2) = sin(tilt);
        R_tilt(2, 0) = 0;   R_tilt(2, 1) = -sin(tilt);  R_tilt(2, 2) = cos(tilt);
        
        Eigen::Matrix3d R_pan;
        R_pan(0, 0) = cos(pan);   R_pan(0, 1) = 0;   R_pan(0, 2) = -sin(pan);
        R_pan(1, 0) = 0;          R_pan(1, 1) = 1;   R_pan(1, 2) = 0;
        R_pan(2, 0) = sin(pan);   R_pan(2, 1) = 0;   R_pan(2, 2) = cos(pan);
        
        m = R_tilt * R_pan;
        return m;
    }
    
    Eigen::Vector2d point2PanTilt(const Eigen::Vector2d& pp,
                                  const Eigen::Vector3d& ptz,
                                  const Eigen::Vector2d& point)
    
    {
        Eigen::Vector2d point_pan_tilt;
        double dx = point.x() - pp.x();
        double dy = point.y() - pp.y();
        double fl = ptz.z();
        double delta_pan = atan2(dx, fl) * 180.0/M_PI;
        double delta_tilt = atan2(dy, fl) * 180.0/M_PI;
        point_pan_tilt[0] = ptz[0] + delta_pan;
        point_pan_tilt[1] = ptz[1] - delta_tilt; // oppositive direction of y
        return point_pan_tilt;
    }
    
    Eigen::Vector2d panTilt2Point(const Eigen::Vector2d& pp,
                                  const Eigen::Vector3d& ptz,
                                  const Eigen::Vector2d& point_pan_tilt)
    {
        double delta_pan  = (point_pan_tilt[0] - ptz[0]) * M_PI/180.0;
        double delta_tilt = (point_pan_tilt[1] - ptz[1]) * M_PI/180.0;
        double fl = ptz[2];
        double delta_x = fl * tan(delta_pan);
        double delta_y = fl * tan(delta_tilt);
        
        Eigen::Vector2d point(pp.x() + delta_x, pp.y() - delta_y); // oppositive direction of y
        return point;
    }
    
    struct SphericalPanTiltFunctor
    {
        typedef double Scalar;
        
        typedef Eigen::VectorXd InputType;
        typedef Eigen::VectorXd ValueType;
        typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;
        
        enum {
            InputsAtCompileTime = Eigen::Dynamic,
            ValuesAtCompileTime = Eigen::Dynamic
        };
        
        const Eigen::Vector2d pp_;
        const vector<Eigen::Vector2d> pan_tilt_; // spherical space
        const vector<Eigen::Vector2d> image_point_;
        
        int m_inputs;
        int m_values;
        
        SphericalPanTiltFunctor(const Eigen::Vector2d & pp,
                                const vector<Eigen::Vector2d> & pan_tilt,
                                const vector<Eigen::Vector2d> & image_point):
        pp_(pp), pan_tilt_(pan_tilt), image_point_(image_point)
        {
            assert(pan_tilt_.size() == image_point_.size());
            m_inputs = 3;
            m_values = (int)pan_tilt.size() * 2;
            assert(m_values >= 4);
        }
        
        
        int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
        {
            double pan  = x[0];
            double tilt = x[1];
            double fl = x[2];
            Eigen::Vector3d ptz(pan, tilt, fl);
            
            // loop each points
            for (int i = 0; i<pan_tilt_.size(); i++) {
                Vector2d cur_pan_tilt = pan_tilt_[i];
                // projection from spherical space to image space
                Vector2d projected_pan_tilt = panTilt2Point(pp_, ptz, cur_pan_tilt);
                fx[2*i + 0] = image_point_[i].x() - projected_pan_tilt.x();
                fx[2*i + 1] = image_point_[i].y() - projected_pan_tilt.y();
            }
            return 0;
        }
        
        double meanReprojectionError(const Eigen::VectorXd& x)
        {
            double pan  = x[0];
            double tilt = x[1];
            double fl = x[2];
            Eigen::Vector3d ptz(pan, tilt, fl);
            
            // loop each points
            double avg_dist = 0.0;
            for (int i = 0; i<pan_tilt_.size(); i++) {
                Vector2d cur_pan_tilt = pan_tilt_[i];
                // projection from spherical space to image space
                Vector2d projected_pan_tilt = panTilt2Point(pp_, ptz, cur_pan_tilt);
                double dist = (image_point_[i] - projected_pan_tilt).norm();
                avg_dist += dist;
            }
            return avg_dist/pan_tilt_.size();
        }

        
        int inputs() const { return m_inputs; }// inputs is the dimension of x.
        int values() const { return m_values; } // "values" is the number of f_i and
    };

    
    double optimizePTZ(const Eigen::Vector2d & pp,
                     const vector<Eigen::Vector2d> & pan_tilt,
                     const vector<Eigen::Vector2d> & image_point,
                     const Vector3d& init_ptz,
                     Vector3d & opt_ptz)
    {
        assert(pan_tilt.size() == image_point.size());
        
        // optimize pan, tilt and focal length
        SphericalPanTiltFunctor opt_functor(pp, pan_tilt, image_point);
        Eigen::NumericalDiff<SphericalPanTiltFunctor> dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<SphericalPanTiltFunctor>, double> lm(dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = 500;
        
        Eigen::VectorXd x(3);
        x[0] = init_ptz[0];
        x[1] = init_ptz[1];
        x[2] = init_ptz[2];
        
        double error = opt_functor.meanReprojectionError(x);
        //printf("reprojection error after: %lf\n", error);
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
        //printf("LMQ status %d\n", status);
        opt_ptz[0] = x[0];
        opt_ptz[1] = x[1];
        opt_ptz[2] = x[2];
        error = opt_functor.meanReprojectionError(x);
        //printf("reprojection error before: %lf\n", error);
        
        return error;
    }

}