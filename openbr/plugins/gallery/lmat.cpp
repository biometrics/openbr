#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup formats
 * \brief Likely matrix format
 *
 * www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class lmatGallery : public Gallery
{
    Q_OBJECT
    QList<cv::Mat> mats;

    ~lmatGallery()
    {
        const likely_const_mat m = likelyFromOpenCVMat(OpenCVUtils::toMatByRow(mats));
        likely_write(m, qPrintable(file.name));
        likely_release_mat(m);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        qFatal("Not supported.");
    }

    void write(const Template &t)
    {
        mats.append(t);
    }
};

BR_REGISTER(Gallery, lmatGallery)

} // namespace br

#include "gallery/lmat.moc"
