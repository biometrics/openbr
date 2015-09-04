#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \brief Perform a number of random transformations to the points in metadata as "Affine_0" and "Affine_1"
 * \author Jordan Cheney \cite jcheney
 * \br_property int numAffines The number of independent random transformations to perform. The result of each transform is stored as its own template in the output TemplateList
 * \br_property float scaleFactor Controls the magnitude of the random changes to the affine points
 * \br_property int maxAngle the maximum angle between the original line between the two affine points and the new line between the points.
 */
class RndAffineTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int numAffines READ get_numAffines WRITE set_numAffines RESET reset_numAffines STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(int maxAngle READ get_maxAngle WRITE set_maxAngle RESET reset_maxAngle STORED false)
    BR_PROPERTY(int, numAffines, 0)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(int, maxAngle, 15)

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            QPointF affine_0 = t.file.get<QPointF>("Affine_0");
            QPointF affine_1 = t.file.get<QPointF>("Affine_1");

            if (affine_0 != QPoint(-1,-1) && affine_1 != QPoint(-1,-1)) {
                // Append the original points
                Template u = t;
                u.file.setPoints(QList<QPointF>() << affine_0 << affine_1);
                u.file.set("Affine_0", affine_0);
                u.file.set("Affine_1", affine_1);
                dst.append(u);

                const double IPD = sqrt(pow(affine_0.x() - affine_1.x(), 2) + pow(affine_0.y() - affine_1.y(), 2));
                if (IPD != 0) {
                    for (int i = 0; i < numAffines; i++) {
                        int angle = (rand() % (2*maxAngle)) - maxAngle;

                        int min = (int)(sqrt(1 / scaleFactor) * IPD);
                        int max = (int)(sqrt(scaleFactor) * IPD);
                        int dx = (rand() % (max - min)) + min;
                        int dy = (dx * sin(angle * CV_PI / 180))/2;

                        QPointF shiftedAffine_0 = QPointF(affine_1.x() - dx, affine_1.y() + dy);

                        Template u = t;
                        u.file.setPoints(QList<QPointF>() << shiftedAffine_0 << affine_1);
                        u.file.set("Affine_0", shiftedAffine_0);
                        u.file.set("Affine_1", affine_1);
                        dst.append(u);
                    }
                }
            }
        }
    }
};

BR_REGISTER(Transform, RndAffineTransform)

} // namespace br

#include "imgproc/rndaffine.moc"
