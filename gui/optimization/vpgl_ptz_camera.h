//
//  vpgl_ptz_camera.h
//  OnlineStereo
//
//  Created by jimmy on 12/29/14.
//  Copyright (c) 2014 Nowhere Planet. All rights reserved.
//

#ifndef __OnlineStereo__vpgl_ptz_camera__
#define __OnlineStereo__vpgl_ptz_camera__

#include <vpgl/vpgl_perspective_camera.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector_fixed.h>


// camera model from "Mimicking Human Camera Operators" from WACV 2015
/*
 The fixed parameter of a PTZ camera is decided by particular setting.
 The following parameters are for WWoS basketball data set
 //estimte fl, pan, tilt from camera
 vgl_point_3d<double> cc(12.9456, -14.8695, 6.21215); //camera center
 vnl_vector_fixed<double, 3> rod;    // 1.58044 -0.118628 0.124857
 rod[0] =    1.58044;
 rod[1] = -0.118628;
 rod[2] =  0.124857;
 R matrix is roughly as
 [1 0 0]
 [0 0 -1]
 [0 1 0]
 
 vnl_vector_fixed<double, 6> coeff;  // 0.570882 0.0310795 -0.533881 -0.000229727 -5.68634e-06 0.000266362
 coeff[0] =  0.570882;
 coeff[1] =  0.0310795;
 coeff[2] = -0.533881;
 coeff[3] = -0.000229727;
 coeff[4] = -5.68634e-06;
 coeff[5] =  0.000266362;
 */

/*
 The following parameters are for WWoS soccer 2014, main PTZ camera
 53.8528 -8.37071 15.0785   1.57061 -0.00440067 0.021745
 vgl_point_3d<double> cc(53.8528, -8.37071, 15.0785); //camera center
 vnl_vector_fixed<double, 3> rod;    //
 rod[0] =   1.57061;
 rod[1] =  -0.00440067;
 rod[2] =   0.021745;
 
 vnl_vector_fixed<double, 6> coeff;
 coeff.fill(0.0);
 */

/*
 The following parameters are for WWoS soccer 2014, left sideview PTZ camera
 CC, Rs is -15.213795 14.944021 5.002864, 1.220866 -1.226907 1.201566
 vgl_point_3d<double> cc(-15.213795, 14.944021, 5.002864);
 vnl_vector_fixed<double, 3> rod(1.220866, -1.226907, 1.201566);
 vnl_vector_fixed<double, 6> coeff;
 coeff.fill(0.0);

 */

class vpgl_ptz_camera:public vpgl_perspective_camera<double>
{
    
protected:
    // all these parameters fixed for a fixed position PTZ camera
    vgl_point_2d<double>        pp_;     // principle point
    vgl_point_3d<double>        cc_;     // rotation center
    vnl_vector_fixed<double, 3> sr_;     // stationary rotation
    vnl_vector_fixed<double, 6> coeff_;  // linear function between projection center and focal length
                                         // all zeros mean rotation center and projection center are the same
    vnl_vector_fixed<double, 3> ptz_;   // pan, tilt and focal length, angles in degree
    
public:
    vpgl_ptz_camera();
    
    // pp: principle point, sr: stationary rotation
    vpgl_ptz_camera(const vgl_point_2d<double> & pp,
                    const vgl_point_3d<double> & rotationCenter,
                    const vnl_vector_fixed<double, 3> & stationayRotationRodrigues,
                    const vnl_vector_fixed<double, 6> & coefficient,
                    double pan, double tilt, double fl);
    
    bool setPTZ(const vgl_point_2d<double> & pp,
                const vgl_point_3d<double> & rotationCenter,
                const vnl_vector_fixed<double, 3> & stationayRotationRodrigues,
                const vnl_vector_fixed<double, 6> & coefficient,
                double pan, double tilt, double fl);
   
    // set PTZ, get camera
    bool setPTZ(const double pan, const double tilt, const double fl);
    // set camera, get PTZ
    bool setCamera(const vpgl_perspective_camera<double> & camera,
                   const vcl_vector<vgl_point_2d<double> > & wld_pts,
                   const vcl_vector<vgl_point_2d<double> > & img_pts);
    
    bool setCamera(const vpgl_perspective_camera<double> & camera,
                   const vcl_vector<vgl_point_3d<double> > & wld_pts,
                   const vcl_vector<vgl_point_2d<double> > & img_pts);
    
    virtual ~vpgl_ptz_camera(){}
    
    double focal_length() const;
    double pan() const;
    double tilt() const;    
    
    vnl_matrix_fixed<double, 3, 3> stationaryRotation() const;
    //  
    
    static void PanYTiltXMatrix(double pan, double tilt, vnl_matrix_fixed<double, 4, 4> &m);
    static void RotationCameraRotationCenterMatrix(const vnl_matrix_fixed<double, 3, 3> &rot, const vgl_point_3d<double> &cameraCenter,
                                                   vnl_matrix_fixed<double, 4, 4> &outMatrix);
    
private:
    // bug, assume the camera coordinate (Z) looking to +Y of world coordinate
    static bool estimateFlPanTiltByFixingModelPositionRotation (const vcl_vector<vgl_point_2d<double> > & wld_pts,
                                                                const vcl_vector<vgl_point_2d<double> > & img_pts,
                                                                const vpgl_perspective_camera<double> & initCamera,
                                                                const vgl_point_3d<double> & cameraCenter, const vnl_vector_fixed<double, 3> & rod,
                                                                const vnl_vector_fixed<double, 6> & coeff, vnl_vector_fixed<double, 3> & flPanTilt,
                                                                vpgl_perspective_camera<double> & estimatedCamera);
    // initial pan, tilt from decompose R_pan_tilt
    static bool estimatePTZByFixingModelPositionRotation (const vcl_vector<vgl_point_3d<double> > & wld_pts,
                                                                const vcl_vector<vgl_point_2d<double> > & img_pts,
                                                                const vpgl_perspective_camera<double> & initCamera,
                                                                const vgl_point_3d<double> & cameraCenter, const vnl_vector_fixed<double, 3> & rod,
                                                                const vnl_vector_fixed<double, 6> & coeff, vnl_vector_fixed<double, 3> & ptz,
                                                                vpgl_perspective_camera<double> & estimatedCamera);
    
    static bool composeCamera(double fl, double pan, double tilt,
                       const vgl_point_3d<double> & rotationCenter,
                       const vnl_vector_fixed<double, 3> & stationayRotationRodrigues,
                       const vnl_vector_fixed<double, 6> & coeff,
                       const vgl_point_2d<double> & principlePoint,
                       vpgl_perspective_camera<double> & camera);
};



#endif /* defined(__OnlineStereo__vpgl_ptz_camera__) */
