#ifndef gl_tolerance_h_
#define gl_tolerance_h_

#include <cmath.h>
#include <limits>

namespace cvx_gl {
    template <typename T>
    class tolerance
    {
    public:
        //! Tolerance for judging 4 points to be coplanar
        static const T point_3d_coplanarity = (T)sqrt(1.0f*std::numeric_limits<T>::epsilon());
        
        //! Tolerance for judging positions to be equal
        static const T position = std::numeric_limits<T>::epsilon();
    };
}


#endif
