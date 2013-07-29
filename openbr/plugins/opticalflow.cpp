#include <opencv2/video/tracking.hpp>
#include "openbr_internal.h"

using namespace cv;

namespace br
{

class OpticalFlowTransform : public UntrainableTransform
{
	Q_OBJECT
	Q_PROPERTY(QDouble pyr_scale READ get_pyr_scale WRITE set_pyr_scale RESET reset_pyr_scale STORE false)
	BR_PROPERTY(double, pyr_scale, 0.5)
	Q_PROPERTY(QString pyr_scale READ get_pyr_scale WRITE set_pyr_scale RESET reset_pyr_scale STORE false)
	BR_PROPERTY(double, pyr_scale, 0.5)

	void project(const Template &src, Template &dst) const
	{
		// get the two images
		// these were the best parameters on KTH
		calcOpticalFlowFarneback(prevImg, nextImg, dst, 0.1, 1, 5, 10, 7, 1.1, 0);
	}
}

} // namespace br

#include 
