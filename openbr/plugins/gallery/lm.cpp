#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

#include <likely.h>
#include <likely/opencv.hpp>

namespace br
{

/*!
 * \ingroup formats
 * \brief Likely matrix format
 *
 * www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class lmGallery : public Gallery
{
    Q_OBJECT
    QList<cv::Mat> mats;

    ~lmGallery()
    {
        if (mats.empty())
            return;
        likely_const_mat m = likelyFromOpenCVMats(mats.toVector().toStdVector());
        if (!likely_write(m, qPrintable(file.name)))
            qFatal("Write failed");
        likely_release_mat(m);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        const likely_const_mat m = likely_read(qPrintable(file.name), likely_file_matrix, likely_void);
        foreach (const cv::Mat &mat, likelyToOpenCVMats(m))
            templates.append(mat);
        likely_release_mat(m);
        return templates;
    }

    void write(const Template &t)
    {
        mats.append(t);
    }
};

BR_REGISTER(Gallery, lmGallery)

} // namespace br

#include "gallery/lm.moc"
