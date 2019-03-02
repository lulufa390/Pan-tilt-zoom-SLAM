
#ifndef __PlanarAlign__PlayField__
#define __PlanarAlign__PlayField__

#include <vector>
#include <vgl/vgl_point_2d.h>
#include <vgl/algo/vgl_h_matrix_2d.h>

using std::vector;

class PlayField
{
public:
	PlayField()
	{

	}
	virtual ~PlayField()
	{

	}

	// points in feet
	// pre-defined point on the playing field
	virtual vector<vgl_point_2d<double> > candindate_points() = 0;

	// pre-defined line intersection on the playing field
	virtual vector<vgl_point_2d<double> > line_intersection_candindate_points() = 0;

	// from image coordinate to world coordinate
	virtual vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p) = 0;  //pixel --> feet
	// from world coordinate to image coordinate
	virtual vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p) = 0;  //feet --> pixel

	// When user click the court image, the location will be away from the exact location.
	// This function returns the closest points in the all candidate points
	// inPt: image coordiante
	// outPt: world coordinate
	// threshold: pixel
	bool find_candinate_point(const vgl_point_2d<double> &inPt,
		vgl_point_2d<double> &outPt,
		double threshold);

	// similar to 'find_candinate_point' but return the line intersection
	bool find_line_intersection_candindate_point(const vgl_point_2d<double> &inPt,
		vgl_point_2d<double> &outPt,
		double threshold);
};

// Disney WWoS playing field
class BasketballPlayField : public PlayField
{
public:
	BasketballPlayField() {}
	~BasketballPlayField() {}

	vector<vgl_point_2d<double> > candindate_points();
	vector<vgl_point_2d<double> > line_intersection_candindate_points();

	// exchange between pixel and world coordinate. This function depends on court image
	vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p);  //pixel --> feet
	vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p);  //feet --> pixel
};

// US high school basketball
class USHSBasketballPlayField : public PlayField
{
public:
	USHSBasketballPlayField();
	~USHSBasketballPlayField();

	vector<vgl_point_2d<double> > candindate_points();
	vector<vgl_point_2d<double> > line_intersection_candindate_points();

	// exchange between pixel and world coordinate. This function depends on court image
	vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p);  //pixel --> feet
	vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p);  //feet --> pixel
};

// Disney soccer field
class WWosSoccerPlayField : public PlayField
{
public:
	WWosSoccerPlayField() {}
	~WWosSoccerPlayField() {}

	vector<vgl_point_2d<double> > candindate_points();  // feet
	vector<vgl_point_2d<double> > line_intersection_candindate_points();
	vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p);  //pixel --> feet
	vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p);  //feet --> pixel

	static vgl_conic<double> getCenterCircle(void);
};

// Soccer World Cup 2014
class Worldcup2014PlayField : public PlayField
{
public:
	Worldcup2014PlayField() {};
	~Worldcup2014PlayField() {};

	vector<vgl_point_2d<double> > candindate_points();  // feet
	vector<vgl_point_2d<double> > line_intersection_candindate_points();
	vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p);  //pixel --> feet
	vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p);  //feet --> pixel

	static vgl_conic<double> getCenterCircle(void);
};

// Volleyball
class VolleyballPlayField : public PlayField
{
public:
	VolleyballPlayField() {}
	~VolleyballPlayField() {}

	vector<vgl_point_2d<double> > candindate_points();     // meter
	vector<vgl_point_2d<double> > line_intersection_candindate_points(); // meter
	vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p);  //pixel --> meter
	vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p);  //meter --> pixel
};

// 200 x 85 feet
class NHLIceHockeyPlayField : public PlayField
{
public:
	NHLIceHockeyPlayField() {}
	~NHLIceHockeyPlayField() {}

	vector<vgl_point_2d<double> > candindate_points();     // feet
	vector<vgl_point_2d<double> > line_intersection_candindate_points(); // feet
	vgl_point_2d<double> image_point_to_world_point(const vgl_point_2d<double> &p);  //pixel --> feet
	vgl_point_2d<double> world_point_to_image_point(const vgl_point_2d<double> &p);  //feet --> pixel

	// get 5 circles
	static vector<vgl_conic<double>> getCircles(void);
};


#endif /* defined(__PlanarAlign__PlayField__) */
