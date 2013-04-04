#ifdef WIN32
#include <windows.h>
#endif

#include <NeoFacePro.h>

#include "openbr_internal.h"
#include "core/resource.h"

using namespace br;

/*!
 * \ingroup initializers
 * \brief Initialize NEC3
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 * \warning Needs a maintainer
 */
class NEC3Initializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        int result = NeoFacePro::Initialize();
        if (result != NFP_SUCCESS) qWarning("NEC3 Initialize error [%d]", result);
        Globals->abbreviations.insert("NEC3", "Open!NEC3Enroll:NEC3Compare");
    }

    void finalize() const
    {
        int result = NeoFacePro::Terminate();
        if (result != NFP_SUCCESS) qWarning("NEC3 Finalize error [%d]", result);
    }
};

BR_REGISTER(Initializer, NEC3Initializer)

/*!
 * \brief NEC3 Context
 * \author Scott Klum \cite sklum
 */

struct NEC3Context
{
    NeoFacePro::CFaceInfo faceInfo;
    NeoFacePro::CFaceFeature faceFeature;

    NEC3Context() {
        faceInfo.SetParamAlgorithm(NFP_ALGORITHM003);
        faceInfo.SetParamEyesRoll(15);
        faceInfo.SetParamEyesMaxWidth(1000);
        faceInfo.SetParamEyesMinWidth(30);
        faceInfo.SetParamMaxFace(1);
        faceInfo.SetParamReliability(0);

        faceFeature.SetParamFeatureType(NFP_FEATURE_S14);
    }
};

/*!
 * \ingroup transforms
 * \brief Enroll a face image in NEC NeoFace 3
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 * \warning Needs a maintainer
 */
class NEC3Enroll : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool detectOnly READ get_detectOnly WRITE set_detectOnly RESET reset_detectOnly STORED false)
    BR_PROPERTY(bool, detectOnly, false)

    Resource<NEC3Context> contexts;
    QSharedPointer<Transform> flip;

    void init()
    {
        contexts.setMaxResources(1);
        flip = QSharedPointer<Transform>(Transform::make("Flip(X)"));
    }

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC3) qFatal("NEC3Enroll requires 8UC3 images.");

        // Bitmaps are stored upside down
        Template flipped;
        flip->project(src, flipped);

        cv::Mat input = flipped.m();
        input = input(cv::Rect(0, 0, (input.cols/4)*4, (input.rows/4)*4)).clone(); // For whatever reason, NEC requires images with 4 pixel alignment...

        BITMAPINFO binfo;
        binfo.bmiHeader.biWidth = input.cols;
        binfo.bmiHeader.biHeight = input.rows;
        binfo.bmiHeader.biBitCount = 24;

        NEC3Context *context = contexts.acquire();
        int result = context->faceInfo.FindFace(binfo, input.data);
        context->faceInfo.LocateEyes();

        if (result == NFP_CANNOT_FIND_FACE) {
            if (Globals->verbose) qDebug("NEC3Enroll face not found for file %s", qPrintable(src.file.flat()));
        } else if (result != NFP_SUCCESS) {
            qWarning("NEC3Enroll FindFace error %d for file %s", result, qPrintable(src.file.flat()));
        }

        QList<QPointF> landmarks;
        for (int i=0; i<context->faceInfo.GetFaceMax(); i++) {
            if (context->faceInfo.SetFaceIndex(i) != NFP_SUCCESS)
                continue;
            POINT right = context->faceInfo.GetRightEye();
            POINT left = context->faceInfo.GetLeftEye();
            landmarks.append(QPointF(right.x, right.y));
            landmarks.append(QPointF(left.x, left.y));

            if (detectOnly) {
                dst += src.m();
            } else {
                result = context->faceFeature.SetFeature(&context->faceInfo);
                if (result != NFP_SUCCESS) {
                    qWarning("NEC3Enroll SetFeature error %d for file %s", result, qPrintable(src.file.flat()));
                    continue;
                }

                void *data;
                long size;
                result = context->faceFeature.Serialize(&data, &size);
                if (result != NFP_SUCCESS) {
                    qWarning("NEC3Enroll Serialize error %d for file %s", result, qPrintable(src.file.flat()));
                    continue;
                }

                dst += cv::Mat(1, size, CV_8UC1, data).clone();
                NeoFacePro::CFaceFeature::FreeSerializeData(data);
            }

            if (src.file.get<bool>("ForceEnrollment", false) && !dst.isEmpty()) break;
        }
        dst.file.appendPoints(landmarks);

        contexts.release(context);

        if (!src.file.get<bool>("enrollAll", false) && dst.isEmpty()) dst += cv::Mat();
    }
};

BR_REGISTER(Transform, NEC3Enroll)

/*!
 * \ingroup distances
 * \brief Compare faces with NEC NeoFace 3 SDK
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 * \warning Needs a maintainer
 */
class NEC3Compare : public Distance
{
    Q_OBJECT

    Resource<NeoFacePro::CVerifier> verifierResource;

    float compare(const Template &a, const Template &b) const
    {
        float score = -std::numeric_limits<float>::max();
        if (a.m().data && b.m().data) {
            NeoFacePro::CVerifier *verifier = verifierResource.acquire();
            int result = verifier->Verify(a.m().data, b.m().data, &score);
            if (result != NFP_SUCCESS) qWarning("NEC3Compare verify error [%d]", result);
            verifierResource.release(verifier);
        }
        return score;
    }
};

BR_REGISTER(Distance, NEC3Compare)

#include "nec3.moc"
