#ifndef _VISUALIZE_
#define _VISUALIZE_

#include "cvimage_view.hpp"

class VisualizeView : public CVImageView
{

public:
	VisualizeView(std::string name);
	~VisualizeView();

	virtual void annotate(){}
	virtual void clearAnnotations(){}
};

#endif
