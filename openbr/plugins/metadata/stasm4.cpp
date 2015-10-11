#include <QString>
#include <stasmcascadeclassifier.h>
#include <stasm_lib.h>
#include <stasm.h>
#include <opencv2/opencv.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/eigenutils.h>

using namespace std;
using namespace cv;

namespace br
{

class StasmResourceMaker : public ResourceMaker<StasmCascadeClassifier>
{
private:
    StasmCascadeClassifier *make() const
    {
        StasmCascadeClassifier *stasmCascade = new StasmCascadeClassifier();
        if (!stasmCascade->load(Globals->sdkPath.toStdString() + "/share/openbr/models/"))
            qFatal("Failed to load Stasm Cascade");
        return stasmCascade;
    }
};

/*!
 * \ingroup initializers
 * \brief Initialize Stasm
 * \author Scott Klum \cite sklum
 */
class StasmInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("RectFromStasmEyes","RectFromPoints([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],0.3,5.3)");
        Globals->abbreviations.insert("RectFromStasmBrow","RectFromPoints([16,17,18,19,20,21,22,23,24,25,26,27],0.15,5)");
        Globals->abbreviations.insert("RectFromStasmPeriocular","RectFromPoints([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,16,17,18,19,20,21,22,23,24,25,26,27]],0.3,5.3)");
        Globals->abbreviations.insert("RectFromStasmBrow","RectFromPoints([16,17,18,19,20,21,22,23,24,25,26,27],0.15,5)");
        Globals->abbreviations.insert("RectFromStasmNose","RectFromPoints([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],0.15,1.15)");
        Globals->abbreviations.insert("RectFromStasmNoseWithBridge", "RectFromPoints([21, 22, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],0.15,.6)");
        Globals->abbreviations.insert("RectFromStasmMouth","RectFromPoints([59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76],0.3,2)");
        Globals->abbreviations.insert("RectFromStasmHair", "RectFromPoints([13,14,15],1.75,1.5)");
        Globals->abbreviations.insert("RectFromStasmJaw", "RectFromPoints([2,3,4,5,6,7,8,9,10],.25,1.6)");
    }
};

BR_REGISTER(Initializer, StasmInitializer)

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */
class StasmTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(bool stasm3Format READ get_stasm3Format WRITE set_stasm3Format RESET reset_stasm3Format STORED false)
    BR_PROPERTY(bool, stasm3Format, false)
    Q_PROPERTY(QList<float> pinPoints READ get_pinPoints WRITE set_pinPoints RESET reset_pinPoints STORED false)
    BR_PROPERTY(QList<float>, pinPoints, QList<float>())
    Q_PROPERTY(QStringList pinLabels READ get_pinLabels WRITE set_pinLabels RESET reset_pinLabels STORED false)
    BR_PROPERTY(QStringList, pinLabels, QStringList())

    Resource<StasmCascadeClassifier> stasmCascadeResource;

    void init()
    {
        if (!stasm_init(qPrintable(Globals->sdkPath + "/share/openbr/models/stasm"), 0)) qFatal("Failed to initalize stasm.");
        stasmCascadeResource.setResourceMaker(new StasmResourceMaker());
    }

    QList<QPointF> convertLandmarks(int nLandmarks, float *landmarks) const
    {
        if (stasm3Format)
            stasm_convert_shape(landmarks, 76);

        QList<QPointF> points;
        for (int i = 0; i < nLandmarks; i++) {
            QPointF point(landmarks[2 * i], landmarks[2 * i + 1]);
            points.append(point);
        }

        return points;
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            Mat stasmSrc(t);
            if (t.m().channels() == 3)
                cvtColor(t, stasmSrc, CV_BGR2GRAY);
            else if (t.m().channels() != 1)
                qFatal("Stasm expects single channel matrices.");

            if (!stasmSrc.isContinuous())
                qFatal("Stasm expects continuous matrix data.");

            int foundFace = 0;
            int nLandmarks = stasm_NLANDMARKS;
            float landmarks[2 * stasm_NLANDMARKS];

            bool searchPinned = false;

            QPointF rightEye, leftEye;
            /* Two use cases are accounted for:
             * 1. Pin eyes without normalization: in this case the string list should contain the KEYS for right then left eyes, respectively.
             * 2. Pin eyes with normalization: in this case the string list should contain the COORDINATES of the right then left eyes, respectively.
             * Currently, we only support normalization with a transformation such that the src file contains Affine_0 and Affine_1.  Checking for
             * these keys prevents us from pinning eyes on a face that wasn't actually transformed (see AffineTransform).
             * If both cases fail, we default to stasm_search_single. */

            if (!pinPoints.isEmpty() && t.file.contains("Affine_0") && t.file.contains("Affine_1")) {
                rightEye = QPointF(pinPoints.at(0), pinPoints.at(1));
                leftEye = QPointF(pinPoints.at(2), pinPoints.at(3));
                searchPinned = true;
            } else if (!pinLabels.isEmpty()) {
                rightEye = t.file.get<QPointF>(pinLabels.at(0), QPointF());
                leftEye = t.file.get<QPointF>(pinLabels.at(1), QPointF());
                searchPinned = true;
            }

            if (searchPinned) {
                float pins[2 * stasm_NLANDMARKS];

                for (int i = 0; i < nLandmarks; i++) {
                    if      (i == 38) /* Stasm Right Eye */ { pins[2*i] = rightEye.x(); pins[2*i+1] = rightEye.y(); }
                    else if (i == 39) /* Stasm Left Eye  */ { pins[2*i] = leftEye.x();  pins[2*i+1] = leftEye.y(); }
                    else { pins[2*i] = 0; pins[2*i+1] = 0; }
                }

                stasm_search_pinned(landmarks, pins, reinterpret_cast<const char*>(stasmSrc.data), stasmSrc.cols, stasmSrc.rows, NULL);

                // The ASM in Stasm is guaranteed to converge in this case
                foundFace = 1;
                Template u(t.file, t.m());
                QList<QPointF> points = convertLandmarks(nLandmarks, landmarks);
                u.file.set("StasmRightEye", points[38]);
                u.file.set("StasmLeftEye", points[39]);
                u.file.appendPoints(points);
                dst.append(u);
            }

            if (!foundFace) {
                StasmCascadeClassifier *stasmCascade = stasmCascadeResource.acquire();
                foundFace = 1;
                stasm::FaceDet detection;
                while (foundFace) {
                    stasm_search_auto(&foundFace, landmarks, reinterpret_cast<const char*>(stasmSrc.data), stasmSrc.cols, stasmSrc.rows, *stasmCascade, detection);
                    if (foundFace) {
                        Template u(t.file, t.m());
                        QList<QPointF> points = convertLandmarks(nLandmarks, landmarks);
                        u.file.set("StasmRightEye", points[38]);
                        u.file.set("StasmLeftEye", points[39]);
                        u.file.appendPoints(points);
                        dst.append(u);
                    }

                    if (!Globals->enrollAll)
                        break;
                }
                stasmCascadeResource.release(stasmCascade);
            }
        }
    }
};

BR_REGISTER(Transform, StasmTransform)

} // namespace br

#include "metadata/stasm4.moc"
