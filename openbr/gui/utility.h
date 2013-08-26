#ifndef BR_UTILITY_H
#define BR_UTILITY_H

#include <QImage>
#include <opencv2/core/core.hpp>

QImage toQImage(const cv::Mat &mat);

#endif // BR_UTILITY_H
