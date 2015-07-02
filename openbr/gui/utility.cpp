#include <limits>
#include <vector>
#include <assert.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <openbr/core/qtutils.h>
#include "utility.h"

using namespace cv;

QImage br::toQImage(const Mat &mat)
{
    // Convert to 8U depth
    Mat mat8u;
    if (mat.depth() != CV_8U) {
        double globalMin = std::numeric_limits<double>::max();
        double globalMax = -std::numeric_limits<double>::max();

        std::vector<Mat> mv;
        split(mat, mv);
        for (size_t i=0; i<mv.size(); i++) {
            double min, max;
            minMaxLoc(mv[i], &min, &max);
            globalMin = std::min(globalMin, min);
            globalMax = std::max(globalMax, max);
        }
        assert(globalMax >= globalMin);

        double range = globalMax - globalMin;
        if (range != 0) {
            double scale = 255 / range;
            convertScaleAbs(mat, mat8u, scale, -(globalMin * scale));
        } else {
            // Monochromatic
            mat8u = Mat(mat.size(), CV_8UC1, Scalar((globalMin+globalMax)/2));
        }
    } else {
        mat8u = mat;
    }

    // Convert to 3 channels
    Mat mat8uc3;
    if      (mat8u.channels() == 4) cvtColor(mat8u, mat8uc3, CV_BGRA2RGB);
    else if (mat8u.channels() == 3) cvtColor(mat8u, mat8uc3, CV_BGR2RGB);
    else if (mat8u.channels() == 1) cvtColor(mat8u, mat8uc3, CV_GRAY2RGB);

    return QImage(mat8uc3.data, mat8uc3.cols, mat8uc3.rows, 3*mat8uc3.cols, QImage::Format_RGB888).copy();
}

QStringList br::getFiles(QDir dir, bool recursive)
{
    dir = QDir(dir.canonicalPath());

    QStringList files;
    foreach (const QString &file, QtUtils::naturalSort(dir.entryList(QDir::Files)))
        files.append(dir.absoluteFilePath(file));

    if (!recursive) return files;

    foreach (const QString &folder, QtUtils::naturalSort(dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))) {
        QDir subdir(dir);
        bool success = subdir.cd(folder); if (!success) qFatal("cd failure.");
        files.append(getFiles(subdir, true));
    }
    return files;
}

QStringList br::getFiles(const QString &regexp)
{
    QFileInfo fileInfo(regexp);
    QDir dir(fileInfo.dir());
    QRegExp re(fileInfo.fileName());
    re.setPatternSyntax(QRegExp::Wildcard);

    QStringList files;
    foreach (const QString &fileName, dir.entryList(QDir::Files))
        if (re.exactMatch(fileName))
            files.append(dir.filePath(fileName));
    return files;
}
