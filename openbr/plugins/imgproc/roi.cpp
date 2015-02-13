#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the rectangular regions of interest.
 * \author Josh Klontz \cite jklontz
 */
class ROITransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        if (!propName.isEmpty()) {
            QRectF rect = src.file.get<QRectF>(propName);
            dst += src.m()(OpenCVUtils::toRect(rect));
        } else if (!src.file.rects().empty()) {
            foreach (const QRectF &rect, src.file.rects())
                dst += src.m()(OpenCVUtils::toRect(rect));
        } else if (src.file.contains(QStringList() << "X" << "Y" << "Width" << "Height")) {
            dst += src.m()(Rect(src.file.get<int>("X"),
                                src.file.get<int>("Y"),
                                src.file.get<int>("Width"),
                                src.file.get<int>("Height")));
        } else {
            dst = src;
            if (Globals->verbose)
                qWarning("No rects present in file.");
        }
        dst.file.clearRects();
    }
};

BR_REGISTER(Transform, ROITransform)

} // namespace br

#include "roi.moc"
