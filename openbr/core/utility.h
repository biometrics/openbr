#ifndef BR_CORE_UTILITY_H
#define BR_CORE_UTILITY_H

#include <QStringList>
#include <QDir>

#include <opencv2/core/core.hpp>
#include <openbr/openbr_export.h>

namespace br
{

BR_EXPORT QStringList getFiles(QDir dir, bool recursive);
BR_EXPORT QStringList getFiles(const QString &regexp);

} // namespace br

#endif // BR_CORE_UTILITY_H
