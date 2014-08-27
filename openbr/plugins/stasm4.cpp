#include <stasm_lib.h>
#include <stasmcascadeclassifier.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/eigenutils.h"
#include <QString>

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
    Q_PROPERTY(QStringList pinEyes READ get_pinEyes WRITE set_pinEyes RESET reset_pinEyes STORED false)
    BR_PROPERTY(QStringList, pinEyes, QStringList())

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
        dst = src;

        StasmCascadeClassifier *stasmCascade = stasmCascadeResource.acquire();

        int foundFace = 0;
        int nLandmarks = stasm_NLANDMARKS;
        float landmarks[2 * stasm_NLANDMARKS];

        if (!pinEyes.isEmpty()) {
            // Two use cases are accounted for:
            // 1. Pin eyes without normalization: in this case the string list should contain the KEYS for right then left eyes, respectively.
            // 2. Pin eyes with normalization: in this case the string list should contain the COORDINATES of the right then left eyes, respectively.
            // If both cases fail, we default to stasm_search_single.

            bool ok = false;
            QPointF rightEye;
            QPointF leftEye;

            if (src.file.contains("Affine_0") && src.file.contains("Affine_1")) {
                rightEye = QtUtils::toPoint(pinEyes.at(0),&ok);
                leftEye = QtUtils::toPoint(pinEyes.at(1),&ok);
            }

            if (!ok) {
                if (src.file.contains(pinEyes.at(0)) && src.file.contains(pinEyes.at(1)))
                {
                    rightEye = src.file.get<QPointF>(pinEyes.at(0), QPointF());
                    leftEye = src.file.get<QPointF>(pinEyes.at(1), QPointF());
                    ok = true;
                }
            }

            float eyes[2 * stasm_NLANDMARKS];

            if (ok) {
                for (int i = 0; i < nLandmarks; i++) {
                    if (i == 38) /*Stasm Right Eye*/ { eyes[2*i] = rightEye.x(); eyes[2*i+1] = rightEye.y(); }
                    else if (i == 39)  /*Stasm Left Eye*/ { eyes[2*i] = leftEye.x(); eyes[2*i+1] = leftEye.y(); }
                    else { eyes[2*i] = 0; eyes[2*i+1] = 0; }
                }
                stasm_search_pinned(landmarks, eyes, reinterpret_cast<const char*>(stasmSrc.data), stasmSrc.cols, stasmSrc.rows, NULL);

                // The ASM in Stasm is guaranteed to converge in this case
                foundFace = 1;
            }
        }

        if (!foundFace) stasm_search_single(&foundFace, landmarks, reinterpret_cast<const char*>(stasmSrc.data), stasmSrc.cols, stasmSrc.rows, *stasmCascade, NULL, NULL);

        if (stasm3Format) {
            nLandmarks = 76;
            stasm_convert_shape(landmarks, nLandmarks);
        }

        stasmCascadeResource.release(stasmCascade);

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

/*!
 * \ingroup transforms
 * \brief Designed for use after eye detection + Stasm, this will
 * revert the detected landmarks to the original coordinate space
 * before affine alignment to the stasm mean shape. The storeAffine
 * parameter must be set to true when calling AffineTransform before this.
 * \author Brendan Klare \cite bklare
 */
class RevertAffineTransform : public UntrainableTransform
{
    Q_OBJECT

private:

    void project(const Template &src, Template &dst) const
    {
        QList<float> paramList = src.file.getList<float>("affineParameters");
        Eigen::MatrixXf points = pointsToMatrix(src.file.points(), true);
        Eigen::MatrixXf affine = Eigen::MatrixXf::Zero(3, 3);
        for (int i = 0, cnt = 0; i < 2; i++)
            for (int j = 0; j < 3; j++, cnt++)
                affine(i, j) = paramList[cnt];
        affine(2, 2) = 1;
        affine = affine.inverse();
        Eigen::MatrixXf affineInv = affine.block(0, 0, 2, 3);
        Eigen::MatrixXf pointsT = points.transpose();
        points =  affineInv * pointsT;
        dst = src;
        dst.file.clearPoints();
        dst.file.setPoints(matrixToPoints(points.transpose()));
    }
};

BR_REGISTER(Transform, RevertAffineTransform)

} // namespace br

#include "stasm4.moc"
