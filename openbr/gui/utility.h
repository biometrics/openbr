#ifndef BR_UTILITY_H
#define BR_UTILITY_H

#include <QImage>
#include <QStringList>
#include <QDir>

#include <opencv2/core/core.hpp>
#include <openbr/openbr_export.h>

namespace br
{

BR_EXPORT QImage toQImage(const cv::Mat &mat);
BR_EXPORT QStringList getFiles(QDir dir, bool recursive);
BR_EXPORT QStringList getFiles(const QString &regexp);

} // namespace br

#endif // BR_UTILITY_H
