#ifndef BR_GUI_UTILITY_H
#define BR_GUI_UTILITY_H

#include <QImage>
#include <QStringList>
#include <QDir>

#include <opencv2/core/core.hpp>
#include <openbr/openbr_export.h>

namespace br
{

BR_EXPORT QImage toQImage(const cv::Mat &mat);

} // namespace br

#endif // BR_GUI_UTILITY_H
