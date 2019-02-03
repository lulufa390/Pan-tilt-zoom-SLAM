//
//  vpgl_ptz_camera.cpp
//  OnlineStereo
//
//  Created by jimmy on 12/29/14.
//  Copyright (c) 2014 Nowhere Planet. All rights reserved.
//
#include "vpgl_ptz_camera.h"
#include <vnl/vnl_least_squares_function.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include <vnl/vnl_inverse.h>


vpgl_ptz_camera::vpgl_ptz_camera()
{
    ptz_ = vnl_vector_fixed<double, 3>(0.0);
    R_.set_identity();
    camera_center_.set( 0.0, 0.0, 0.0 );
    recompute_matrix();
}

vpgl_ptz_camera::vpgl_ptz_camera(const vgl_point_2d<double> & pp,
                                 const vgl_point_3d<double> & rotationCenter,
                                 const vnl_vector_fixed<double, 3> & stationaryRotation,
                                 const vnl_vector_fixed<double, 6> & coefficient, double pan, double tilt, double fl)
{
    ptz_[0] = pan;
    ptz_[1] = tilt;
    ptz_[2] = fl;
  
    // store fixed parameter
    pp_ = pp;
    cc_ = rotationCenter;
    sr_ = stationaryRotation;
    coeff_ = coefficient;
    
    // compose all paramerter to a projective camera
    vpgl_perspective_camera<double> camera;
    bool isCamera = vpgl_ptz_camera::composeCamera(fl, pan, tilt, cc_, sr_, coeff_, pp_, camera);
    assert(isCamera);
    
    K_ = camera.get_calibration();
    camera_center_ = camera.get_camera_center();
    R_ = camera.get_rotation();
    recompute_matrix();
}

bool vpgl_ptz_camera::setPTZ(const vgl_point_2d<double> & pp,
                             const vgl_point_3d<double> & rotationCenter,
                             const vnl_vector_fixed<double, 3> & stationaryRotation,
                             const vnl_vector_fixed<double, 6> & coefficient,
                             double pan, double tilt, double fl)
{
    ptz_[0] = pan;
    ptz_[1] = tilt;
    ptz_[2] = fl;
    
    // store fixed parameter
    pp_ = pp;
    cc_ = rotationCenter;
    sr_ = stationaryRotation;
    coeff_ = coefficient;
    
    // compose all paramerter to a projective camera
    vpgl_perspective_camera<double> camera;
    bool isCamera = vpgl_ptz_camera::composeCamera(fl, pan, tilt, cc_, sr_, coeff_, pp_, camera);
    if (isCamera) {
        K_ = camera.get_calibration();
        camera_center_ = camera.get_camera_center();
        R_ = camera.get_rotation();
        recompute_matrix();
    }
    return isCamera;
}

bool vpgl_ptz_camera::setPTZ(const double pan, const double tilt, const double fl)
{
    ptz_[0] = pan;
    ptz_[1] = tilt;
    ptz_[2] = fl;
    
    // compose all paramerter to a projective camera
    vpgl_perspective_camera<double> camera;
    bool isCamera = vpgl_ptz_camera::composeCamera(fl, pan, tilt, cc_, sr_, coeff_, pp_, camera);
    if(!isCamera){
        return false;
    }
    
    K_ = camera.get_calibration();
    camera_center_ = camera.get_camera_center();
    R_ = camera.get_rotation();
    recompute_matrix();
    return true;
}

bool vpgl_ptz_camera::setCamera(const vpgl_perspective_camera<double> & camera,
                                const vcl_vector<vgl_point_2d<double> > & wld_pts,
                                const vcl_vector<vgl_point_2d<double> > & img_pts)
{
    K_ = camera.get_calibration();
    camera_center_ = camera.get_camera_center();
    R_ = camera.get_rotation();
    recompute_matrix();
    
    vpgl_perspective_camera<double> estimatedCamera;
    vnl_vector_fixed<double, 3> fl_pt;
    bool isEsimated = vpgl_ptz_camera::estimateFlPanTiltByFixingModelPositionRotation(wld_pts, img_pts, camera, camera_center_, sr_, coeff_, fl_pt, estimatedCamera);
    ptz_[0] = fl_pt[1];
    ptz_[1] = fl_pt[2];
    ptz_[2] = fl_pt[0];
    return isEsimated;
}

bool vpgl_ptz_camera::setCamera(const vpgl_perspective_camera<double> & camera,
                                const vcl_vector<vgl_point_3d<double> > & wld_pts,
                                const vcl_vector<vgl_point_2d<double> > & img_pts)
{
    K_ = camera.get_calibration();
    camera_center_ = camera.get_camera_center();
    R_ = camera.get_rotation();
    recompute_matrix();
    
    vpgl_perspective_camera<double> estimatedCamera;
    bool isEsimated = vpgl_ptz_camera::estimatePTZByFixingModelPositionRotation(wld_pts, img_pts, camera, camera_center_, sr_, coeff_, ptz_, estimatedCamera);
    return isEsimated;
}

double vpgl_ptz_camera::focal_length() const
{
    return ptz_[2];
}
double  vpgl_ptz_camera::pan()const
{
    return ptz_[0];
}
double vpgl_ptz_camera::tilt() const
{
    return ptz_[1];
}

vnl_matrix_fixed<double, 3, 3>  vpgl_ptz_camera::stationaryRotation() const
{
    vgl_rotation_3d<double> R(sr_);
    return R.as_matrix();
}


class estimatePanTiltByFixsingModelPositionRotationResidual: public vnl_least_squares_function
{
protected:
    const vcl_vector<vgl_point_2d<double> > & wld_pts_;
    const vcl_vector<vgl_point_2d<double> > & img_pts_;
    const vgl_point_2d<double> principlePoint_;
    const vgl_point_3d<double> cameraCenter_;
    const vnl_vector_fixed<double, 3> rod_; //rodrigue angle of model rotation
    const vnl_vector_fixed<double, 6> coeff_;
    
public:
    estimatePanTiltByFixsingModelPositionRotationResidual(const vcl_vector<vgl_point_2d<double> > & wld_pts,
                                                          const vcl_vector<vgl_point_2d<double> > & img_pts,
                                                          const vgl_point_2d<double> & pp,
                                                          const vgl_point_3d<double> & cc, const vnl_vector_fixed<double, 3> & rod,
                                                          const vnl_vector_fixed<double, 6> & coeff,
                                                          int pts_num):
    vnl_least_squares_function(3, 2 * pts_num, no_gradient),
    wld_pts_(wld_pts),
    img_pts_(img_pts),
    principlePoint_(pp),
    cameraCenter_(cc),
    rod_(rod),
    coeff_(coeff)
    {
        assert(wld_pts.size() >= 4);
        assert(img_pts.size() >= 4);
        assert(wld_pts.size() == img_pts.size());
    }
    
    void f(vnl_vector<double> const &x, vnl_vector<double> &fx)
    {
        double fl   = x[0];
        double pan  = x[1];
        double tilt = x[2];
        double c1 = coeff_[0];
        double c2 = coeff_[1];
        double c3 = coeff_[2];
        double c4 = coeff_[3];
        double c5 = coeff_[4];
        double c6 = coeff_[5];
        
        vgl_rotation_3d<double> Rs(rod_);  // model rotation
        
        vpgl_calibration_matrix<double> K(fl, principlePoint_);
        vnl_matrix_fixed<double, 3, 4> C;
        C.set_identity();
        C(0, 3) = - (c1 + c4 * fl);
        C(1, 3) = - (c2 + c5 * fl);
        C(2, 3) = - (c3 + c6 * fl);
        
        vnl_matrix_fixed<double, 4, 4> Q;  //rotation from pan tilt angle
        vpgl_ptz_camera::PanYTiltXMatrix(pan, tilt, Q);
        
        vnl_matrix_fixed<double, 4, 4> RSD;
        vpgl_ptz_camera::RotationCameraRotationCenterMatrix(Rs.as_matrix(), cameraCenter_,  RSD);
        
        vnl_matrix_fixed<double, 3, 4> P = K.get_matrix() * C * Q * RSD;
        vpgl_proj_camera<double> camera(P);
        
        // loop each points
        int idx = 0;
        for (int i = 0; i<wld_pts_.size(); i++) {
            vgl_point_2d<double> p = wld_pts_[i];
            vgl_point_2d<double> q = (vgl_point_2d<double>)(camera.project(vgl_point_3d<double>(p.x(), p.y(), 0.0)));
            
            fx[idx] = img_pts_[i].x() - q.x();
            idx++;
            fx[idx] = img_pts_[i].y() - q.y();
            idx++;
        }
    }
    
    void getProjection(vnl_vector<double> const &x, vpgl_proj_camera<double> & projection)
    {
        double fl   = x[0];
        double pan  = x[1];
        double tilt = x[2];
        double c1 = coeff_[0];
        double c2 = coeff_[1];
        double c3 = coeff_[2];
        double c4 = coeff_[3];
        double c5 = coeff_[4];
        double c6 = coeff_[5];
        
        vgl_rotation_3d<double> Rs(rod_);  // model rotation
        
        vpgl_calibration_matrix<double> K(fl, principlePoint_);
        vnl_matrix_fixed<double, 3, 4> C;
        C.set_identity();
        C(0, 3) = - (c1 + c4 * fl);
        C(1, 3) = - (c2 + c5 * fl);
        C(2, 3) = - (c3 + c6 * fl);
        
        vnl_matrix_fixed<double, 4, 4> Q;  //rotation from pan tilt angle
        vpgl_ptz_camera::PanYTiltXMatrix(pan, tilt, Q);
        
        vnl_matrix_fixed<double, 4, 4> RSD;
        vpgl_ptz_camera::RotationCameraRotationCenterMatrix(Rs.as_matrix(), cameraCenter_,  RSD);
        
        vnl_matrix_fixed<double, 3, 4> P = K.get_matrix() * C * Q * RSD;
        projection = vpgl_proj_camera<double>(P);
    }
};

bool vpgl_ptz_camera::estimateFlPanTiltByFixingModelPositionRotation (const vcl_vector<vgl_point_2d<double> > & wld_pts,
                                                                      const vcl_vector<vgl_point_2d<double> > & img_pts,
                                                                      const vpgl_perspective_camera<double> & initCamera,
                                                           const vgl_point_3d<double> & cameraCenter, const vnl_vector_fixed<double, 3> & rod,
                                                           const vnl_vector_fixed<double, 6> & coeff, vnl_vector_fixed<double, 3> & flPanTilt,
                                                           vpgl_perspective_camera<double> & estimatedCamera)
{
    assert(wld_pts.size() == img_pts.size());
    assert(wld_pts.size() >= 4);
    assert(img_pts.size() >= 4);
    
    estimatePanTiltByFixsingModelPositionRotationResidual residual(wld_pts, img_pts, initCamera.get_calibration().principal_point(),
                                                                   cameraCenter, rod, coeff, (int)img_pts.size());
    
    // init values
    vnl_vector<double> x(3, 0.0);
    x[0] = initCamera.get_calibration().get_matrix()[0][0];
    double wx = initCamera.get_rotation().as_matrix()[2][0];
    double wy = initCamera.get_rotation().as_matrix()[2][1];
    double wz = initCamera.get_rotation().as_matrix()[2][2];
    double pan  = atan2(wx, wy) * 180.0 /vnl_math::pi;
    double tilt = atan2(wz, wy) * 180.0 /vnl_math::pi;
    x[1] = pan;
    x[2] = tilt;
    
    vnl_levenberg_marquardt lmq(residual);
    
    bool isMinimized = lmq.minimize(x);
    if (!isMinimized) {
        vcl_cerr<<"Error: minimization failed.\n";
        lmq.diagnose_outcome();
        return false;
    }
    //   lmq.diagnose_outcome();
    
    flPanTilt[0] = x[0];
    flPanTilt[1] = x[1];
    flPanTilt[2] = x[2];
    
    vpgl_proj_camera<double> projection;
    residual.getProjection(x, projection);
    
    return vpgl_perspective_decomposition(projection.get_matrix(), estimatedCamera);
}

class estimatePTZByFixingModelPositionRotationResidual: public vnl_least_squares_function
{
protected:
    const vcl_vector<vgl_point_3d<double> > & wld_pts_;
    const vcl_vector<vgl_point_2d<double> > & img_pts_;
    const vgl_point_2d<double> principlePoint_;
    const vgl_point_3d<double> cameraCenter_;
    const vnl_vector_fixed<double, 3> rod_; //rodrigue angle of model rotation
    const vnl_vector_fixed<double, 6> coeff_;
    
public:
    estimatePTZByFixingModelPositionRotationResidual(const vcl_vector<vgl_point_3d<double> > & wld_pts,
                                                          const vcl_vector<vgl_point_2d<double> > & img_pts,
                                                          const vgl_point_2d<double> & pp,
                                                          const vgl_point_3d<double> & cc, const vnl_vector_fixed<double, 3> & rod,
                                                          const vnl_vector_fixed<double, 6> & coeff,
                                                          int pts_num):
    vnl_least_squares_function(3, 2 * pts_num, no_gradient),
    wld_pts_(wld_pts),
    img_pts_(img_pts),
    principlePoint_(pp),
    cameraCenter_(cc),
    rod_(rod),
    coeff_(coeff)
    {
        assert(wld_pts.size() >= 2);
        assert(img_pts.size() >= 2);
        assert(wld_pts.size() == img_pts.size());
    }
    
    void f(vnl_vector<double> const &x, vnl_vector<double> &fx)
    {
        double pan  = x[0];
        double tilt = x[1];
        double fl   = x[2];
        double c1 = coeff_[0];
        double c2 = coeff_[1];
        double c3 = coeff_[2];
        double c4 = coeff_[3];
        double c5 = coeff_[4];
        double c6 = coeff_[5];
        
        vgl_rotation_3d<double> Rs(rod_);  // model rotation
        
        vpgl_calibration_matrix<double> K(fl, principlePoint_);
        vnl_matrix_fixed<double, 3, 4> C;
        C.set_identity();
        C(0, 3) = - (c1 + c4 * fl);
        C(1, 3) = - (c2 + c5 * fl);
        C(2, 3) = - (c3 + c6 * fl);
        
        vnl_matrix_fixed<double, 4, 4> Q;  //rotation from pan tilt angle
        vpgl_ptz_camera::PanYTiltXMatrix(pan, tilt, Q);
        
        vnl_matrix_fixed<double, 4, 4> RSD;
        vpgl_ptz_camera::RotationCameraRotationCenterMatrix(Rs.as_matrix(), cameraCenter_,  RSD);
        
        vnl_matrix_fixed<double, 3, 4> P = K.get_matrix() * C * Q * RSD;
        vpgl_proj_camera<double> camera(P);
        
        // loop each points
        int idx = 0;
        for (int i = 0; i<wld_pts_.size(); i++) {
            vgl_point_3d<double> p = wld_pts_[i];
            vgl_point_2d<double> q = camera.project(p);
            
            fx[idx] = img_pts_[i].x() - q.x();
            idx++;
            fx[idx] = img_pts_[i].y() - q.y();
            idx++;
        }
    }
    
    void getProjection(vnl_vector<double> const &x, vpgl_proj_camera<double> & projection)
    {
        double pan  = x[0];
        double tilt = x[1];
        double fl   = x[2];
        double c1 = coeff_[0];
        double c2 = coeff_[1];
        double c3 = coeff_[2];
        double c4 = coeff_[3];
        double c5 = coeff_[4];
        double c6 = coeff_[5];
        
        vgl_rotation_3d<double> Rs(rod_);  // model rotation
        
        vpgl_calibration_matrix<double> K(fl, principlePoint_);
        vnl_matrix_fixed<double, 3, 4> C;
        C.set_identity();
        C(0, 3) = - (c1 + c4 * fl);
        C(1, 3) = - (c2 + c5 * fl);
        C(2, 3) = - (c3 + c6 * fl);
        
        vnl_matrix_fixed<double, 4, 4> Q;  //rotation from pan tilt angle
        vpgl_ptz_camera::PanYTiltXMatrix(pan, tilt, Q);
        
        vnl_matrix_fixed<double, 4, 4> RSD;
        vpgl_ptz_camera::RotationCameraRotationCenterMatrix(Rs.as_matrix(), cameraCenter_,  RSD);
        
        vnl_matrix_fixed<double, 3, 4> P = K.get_matrix() * C * Q * RSD;
        projection = vpgl_proj_camera<double>(P);
    }
};


bool vpgl_ptz_camera::estimatePTZByFixingModelPositionRotation (const vcl_vector<vgl_point_3d<double> > & wld_pts,
                                               const vcl_vector<vgl_point_2d<double> > & img_pts,
                                               const vpgl_perspective_camera<double> & initCamera,
                                               const vgl_point_3d<double> & cameraCenter, const vnl_vector_fixed<double, 3> & rod,
                                               const vnl_vector_fixed<double, 6> & coeff, vnl_vector_fixed<double, 3> & ptz,
                                               vpgl_perspective_camera<double> & estimatedCamera)
{
    assert(wld_pts.size() == img_pts.size());
    assert(wld_pts.size() >= 2);
    assert(img_pts.size() >= 2);
    
    vgl_rotation_3d<double> Rs(rod);
    vnl_matrix_fixed<double, 3, 3> Rs_inv = vnl_inverse(Rs.as_matrix());
    
    // init values
    vnl_vector<double> x(3, 0.0);
    vnl_matrix_fixed<double, 3, 3> R_pan_tilt = initCamera.get_rotation().as_matrix() * Rs_inv;
    double cos_pan = R_pan_tilt(0, 0);
    double sin_pan = -R_pan_tilt(0, 2);
    double cos_tilt = R_pan_tilt(1, 1);
    double sin_tilt = -R_pan_tilt(2, 1);
    double pan  = atan2(sin_pan, cos_pan) * 180.0 /vnl_math::pi;
    double tilt = atan2(sin_tilt, cos_tilt) * 180.0 /vnl_math::pi;
    x[0] = pan;
    x[1] = tilt;
    x[2] = initCamera.get_calibration().get_matrix()[0][0];
    
    estimatePTZByFixingModelPositionRotationResidual residual(wld_pts, img_pts, initCamera.get_calibration().principal_point(),
                                                                   cameraCenter, rod, coeff, (int)img_pts.size());
    
    vnl_levenberg_marquardt lmq(residual);
    
    bool isMinimized = lmq.minimize(x);
    if (!isMinimized) {
        vcl_cerr<<"Error: minimization failed.\n";
        lmq.diagnose_outcome();
        return false;
    }
    // lmq.diagnose_outcome();
    ptz[0] = x[0];
    ptz[1] = x[1];
    ptz[2] = x[2];    
    vpgl_proj_camera<double> projection;
    residual.getProjection(x, projection);
    
    return vpgl_perspective_decomposition(projection.get_matrix(), estimatedCamera);
}


void vpgl_ptz_camera::PanYTiltXMatrix(double pan, double tilt, vnl_matrix_fixed<double, 4, 4> &m)
{
    vnl_matrix<double> m33(3, 3);
	
    pan *= vnl_math::pi / 180.0;
    tilt *= vnl_math::pi / 180.0;
    
    vnl_matrix_fixed<double, 3, 3> R_tilt;
    R_tilt[0][0] = 1;   R_tilt[0][1] = 0;          R_tilt[0][2] = 0;
    R_tilt[1][0] = 0;   R_tilt[1][1] = cos(tilt);  R_tilt[1][2] = sin(tilt);
    R_tilt[2][0] = 0;   R_tilt[2][1] = -sin(tilt);  R_tilt[2][2] = cos(tilt);
    
    vnl_matrix_fixed<double, 3, 3> R_pan;
    R_pan[0][0] = cos(pan);   R_pan[0][1] = 0;   R_pan[0][2] = -sin(pan);
    R_pan[1][0] = 0;          R_pan[1][1] = 1;   R_pan[1][2] = 0;
    R_pan[2][0] = sin(pan);  R_pan[2][1] = 0;   R_pan[2][2] = cos(pan);
    
    m33 = R_tilt * R_pan;
    
    m.set_identity();
    m.update(m33);
}

void vpgl_ptz_camera::RotationCameraRotationCenterMatrix(const vnl_matrix_fixed<double, 3, 3> &rot,
                                                         const vgl_point_3d<double> &cameraCenter,
                                                         vnl_matrix_fixed<double, 4, 4> &outMatrix)
{
    vnl_matrix_fixed<double, 3, 4> translation;
    translation.set_identity();
    translation[0][3] = -cameraCenter.x();
    translation[1][3] = -cameraCenter.y();
    translation[2][3] = -cameraCenter.z();
    
    vnl_matrix_fixed<double, 3, 4> rot_tras_34 = rot * translation;
    
    outMatrix.set_identity();
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<4; j++) {
            outMatrix[i][j] = rot_tras_34[i][j];
        }
    }
}



bool vpgl_ptz_camera::composeCamera(double fl, double pan, double tilt,
                                    const vgl_point_3d<double> & rotationCenter,
                                    const vnl_vector_fixed<double, 3> & stationayRotationRodrigues,
                                    const vnl_vector_fixed<double, 6> & coeff,
                                    const vgl_point_2d<double> & principlePoint,
                                    vpgl_perspective_camera<double> & camera)
{
    double c1 = coeff[0];
    double c2 = coeff[1];
    double c3 = coeff[2];
    double c4 = coeff[3];
    double c5 = coeff[4];
    double c6 = coeff[5];
    
    vgl_rotation_3d<double> Rs(stationayRotationRodrigues);  // model rotation
    
    vpgl_calibration_matrix<double> K(fl, principlePoint);
    vnl_matrix_fixed<double, 3, 4> C;
    C.set_identity();
    C(0, 3) = - (c1 + c4 * fl);
    C(1, 3) = - (c2 + c5 * fl);
    C(2, 3) = - (c3 + c6 * fl);
    
    vnl_matrix_fixed<double, 4, 4> Q;  //rotation from pan tilt angle
    vpgl_ptz_camera::PanYTiltXMatrix(pan, tilt, Q);
    
    vnl_matrix_fixed<double, 4, 4> RSD;
    vpgl_ptz_camera::RotationCameraRotationCenterMatrix(Rs.as_matrix(), rotationCenter,  RSD);
    
    vnl_matrix_fixed<double, 3, 4> P = K.get_matrix() * C * Q * RSD;
    
    bool isDecompose = vpgl_perspective_decomposition(P, camera);
    if (!isDecompose) {
        return false;
    }
    
    double fl_n = camera.get_calibration().get_matrix()[0][0];
    vgl_point_2d<double> pp = camera.get_calibration().principal_point();
    vpgl_calibration_matrix<double> KK(fl_n, pp);
    camera.set_calibration(KK);
    return true;
}

