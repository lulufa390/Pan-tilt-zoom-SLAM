#ifndef __PlanarAlign__NHL_court__
#define __PlanarAlign__NHL_court__

#include <iostream>
#include <vector>
#include <vgl/vgl_line_segment_2d.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

using std::vector;
using cv::Mat;

// NHL ice hockey rink for camera visualization
//

class NHLCourt {
private:
    vector<vgl_line_segment_2d< double > > markings_;  // line segments and circles
    vector<vgl_point_2d<double> > calib_points_;       // intersection points in the rink
public:
    // read court configuration from a file
    NHLCourt(const char *court_file = "./resource/ice_hockey_model.txt");
    ~NHLCourt(){}
    
    // overlay projected line segment on the image
    void overlayLines(const vpgl_perspective_camera<double> & camera, Mat & image) const;
    
    void overlayPoints(const vpgl_perspective_camera<double> & camera, Mat & image) const;
};

#endif /* defined(__PlanarAlign__NHL_court__) */
