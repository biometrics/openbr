#ifndef UTILITY_H
#define UTILITY_H

#include <QImage>
#include <opencv2/core/core.hpp>

QImage toQImage(const cv::Mat &mat);

#endif // UTILITY_H
