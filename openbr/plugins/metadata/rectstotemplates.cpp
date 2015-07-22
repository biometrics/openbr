#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief For each rectangle bounding box in src, a new Template is created.
 * \author Brendan Klare \cite bklare
 */
class RectsToTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
        Template tOut(src.file);
        QList<float> confidences = src.file.getList<float>("Confidences");
        QList<QRectF> rects = src.file.rects();
        for (int i = 0; i < rects.size(); i++) {
            cv::Mat m(src, OpenCVUtils::toRect(rects[i]));
            Template t(src.file, m);
            t.file.set("Confidence", confidences[i]);
            t.file.clearRects();
            tOut << t;
        }
        dst = tOut;
    }
};

BR_REGISTER(Transform, RectsToTemplatesTransform)

} // namespace br

#include "metadata/rectstotemplates.moc"
