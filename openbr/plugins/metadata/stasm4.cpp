#include <QString>
#include <stasm_lib.h>
#include <stasmcascadeclassifier.h>
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
class StasmTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(bool stasm3Format READ get_stasm3Format WRITE set_stasm3Format RESET reset_stasm3Format STORED false)
    BR_PROPERTY(bool, stasm3Format, false)
    Q_PROPERTY(bool clearLandmarks READ get_clearLandmarks WRITE set_clearLandmarks RESET reset_clearLandmarks STORED false)
    BR_PROPERTY(bool, clearLandmarks, false)
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

    void project(const Template &src, Template &dst) const
    {
        Mat stasmSrc(src);
        if (src.m().channels() == 3)
            cvtColor(src, stasmSrc, CV_BGR2GRAY);
        else if (src.m().channels() != 1)
            qFatal("Stasm expects single channel matrices.");

        if (!stasmSrc.isContinuous())
            qFatal("Stasm expects continuous matrix data.");
        dst = src;

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

        if (!pinPoints.isEmpty() && src.file.contains("Affine_0") && src.file.contains("Affine_1")) {
            rightEye = QPointF(pinPoints.at(0), pinPoints.at(1));
            leftEye = QPointF(pinPoints.at(2), pinPoints.at(3));
            searchPinned = true;
        } else if (!pinLabels.isEmpty()) {
            rightEye = src.file.get<QPointF>(pinLabels.at(0), QPointF());
            leftEye = src.file.get<QPointF>(pinLabels.at(1), QPointF());
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
        }

        if (!foundFace) {
            StasmCascadeClassifier *stasmCascade = stasmCascadeResource.acquire();
            stasm_search_single(&foundFace, landmarks, reinterpret_cast<const char*>(stasmSrc.data), stasmSrc.cols, stasmSrc.rows, *stasmCascade, NULL, NULL);
            stasmCascadeResource.release(stasmCascade);
        }

        if (stasm3Format) {
            nLandmarks = 76;
            stasm_convert_shape(landmarks, nLandmarks);
        }

        // For convenience, if these are the only points/rects we want to deal with as the algorithm progresses
        if (clearLandmarks) {
            dst.file.clearPoints();
            dst.file.clearRects();
        }

        if (!foundFace) {
            if (Globals->verbose) qWarning("No face found in %s.", qPrintable(src.file.fileName()));
            dst.file.fte = true;
        } else {
            QList<QPointF> points;
            for (int i = 0; i < nLandmarks; i++) {
                QPointF point(landmarks[2 * i], landmarks[2 * i + 1]);
                points.append(point);
            }
            dst.file.set("StasmRightEye", points[38]);
            dst.file.set("StasmLeftEye", points[39]);
            dst.file.appendPoints(points);
        }
    }
};

BR_REGISTER(Transform, StasmTransform)

} // namespace br

#include "metadata/stasm4.moc"
