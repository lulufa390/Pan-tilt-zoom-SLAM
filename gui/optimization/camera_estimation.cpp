//
//  camera_estimation.cpp
//  Annotation
//
//  Created by jimmy on 2019-01-27.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "camera_estimation.hpp"
#include <vpgl/algo/vpgl_calibration_matrix_compute.h>
#include <vpgl/algo/vpgl_camera_compute.h>
#include <vnl/vnl_least_squares_function.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>


namespace cvx {
    bool init_calib(const vector<vgl_point_2d<double> > &wld_pts,
                                   const vector<vgl_point_2d<double> > &img_pts,
                                   const vgl_point_2d<double> &principlePoint,
                                   vpgl_perspective_camera<double> &camera)
    {
        if (wld_pts.size() < 4 && img_pts.size() < 4) {
            return false;
        }
        if (wld_pts.size() != img_pts.size()) {
            return false;
        }
        assert(wld_pts.size() >= 4 && img_pts.size() >= 4);
        assert(wld_pts.size() == img_pts.size());
        
        vpgl_calibration_matrix<double> K;
        if (vpgl_calibration_matrix_compute::natural(img_pts, wld_pts, principlePoint, K) == false) {
            std::cerr<<"Failed to compute K"<<std::endl;
            std::cerr<<"Default principle point: "<<principlePoint<<std::endl;
            return false;
        }
        
        camera.set_calibration(K);
        
        // vpgl_perspective_camera_compute_positiveZ
        if (vpgl_perspective_camera_compute::compute(img_pts, wld_pts, camera) == false) {
            std::cerr<<"Failed to computer R, C"<<std::endl;
            return false;
        }
        return true;
    }
    
    
    class optimize_perspective_camera_residual:public vnl_least_squares_function
    {
    protected:
        const std::vector<vgl_point_2d<double> > wldPts_;
        const std::vector<vgl_point_2d<double> > imgPts_;
        const vgl_point_2d<double> principlePoint_;
        
    public:
        optimize_perspective_camera_residual(const std::vector<vgl_point_2d<double> > & wldPts,
                                             const std::vector<vgl_point_2d<double> > & imgPts,
                                             const vgl_point_2d<double> & pp):
        vnl_least_squares_function(7, (unsigned int)(wldPts.size()) * 2, no_gradient),
        wldPts_(wldPts),
        imgPts_(imgPts),
        principlePoint_(pp)
        {
            assert(wldPts.size() == imgPts.size());
            assert(wldPts.size() >= 4);
        }
        
        void f(vnl_vector<double> const &x, vnl_vector<double> &fx)
        {
            //focal length, Rxyz, Camera_center_xyz
            vpgl_calibration_matrix<double> K(x[0], principlePoint_);
            
            vnl_vector_fixed<double, 3> rod(x[1], x[2], x[3]);
            vgl_rotation_3d<double>  R(rod);
            vgl_point_3d<double> cc(x[4], x[5], x[6]);  //camera center
            
            vpgl_perspective_camera<double> camera;
            camera.set_calibration(K);
            camera.set_rotation(R);
            camera.set_camera_center(cc);
            
            //loop all points
            int idx = 0;
            for (int i = 0; i<wldPts_.size(); i++) {
                vgl_point_3d<double> p(wldPts_[i].x(), wldPts_[i].y(), 0);
                vgl_point_2d<double> proj_p = (vgl_point_2d<double>)camera.project(p);
                
                fx[idx] = imgPts_[i].x() - proj_p.x();
                idx++;
                fx[idx] = imgPts_[i].y() - proj_p.y();
                idx++;
            }
        }
        
        void getCamera(vnl_vector<double> const &x, vpgl_perspective_camera<double> &camera)
        {
            
            vpgl_calibration_matrix<double> K(x[0], principlePoint_);
            
            vnl_vector_fixed<double, 3> rod(x[1], x[2], x[3]);
            vgl_rotation_3d<double>  R(rod);
            vgl_point_3d<double> camera_center(x[4], x[5], x[6]);
            
            camera.set_calibration(K);
            camera.set_rotation(R);
            camera.set_camera_center(camera_center);
        }
        
    };
    
    
    bool optimize_perspective_camera(const std::vector<vgl_point_2d<double> > & wld_pts,
                                    const std::vector<vgl_point_2d<double> > & img_pts,
                                    const vpgl_perspective_camera<double> &init_camera,
                                    vpgl_perspective_camera<double> & final_camera)
    {
        assert(wld_pts.size() == img_pts.size());
        assert(wld_pts.size() >= 4);
        
        optimize_perspective_camera_residual residual(wld_pts, img_pts,
                                                      init_camera.get_calibration().principal_point());
        
        vnl_vector<double> x(7, 0);
        x[0] = init_camera.get_calibration().get_matrix()[0][0];
        x[1] = init_camera.get_rotation().as_rodrigues()[0];
        x[2] = init_camera.get_rotation().as_rodrigues()[1];
        x[3] = init_camera.get_rotation().as_rodrigues()[2];
        x[4] = init_camera.camera_center().x();
        x[5] = init_camera.camera_center().y();
        x[6] = init_camera.camera_center().z();
        
        vnl_levenberg_marquardt lmq(residual);
        
        bool isMinimied = lmq.minimize(x);
        if (!isMinimied) {
            std::cerr<<"Error: perspective camera optimize not converge.\n";
            lmq.diagnose_outcome();
            return false;
        }
        lmq.diagnose_outcome();
        
        //    lmq.diagnose_outcome();
        residual.getCamera(x, final_camera);
        return true;
    }


    
    
}









