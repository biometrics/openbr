#include <NeoFacePro.h>
#include <mm_plugin.h>

#include "common/resource.h"

using namespace mm;

/*!
 * \ingroup initializers
 * \brief Initialize NEC3
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer
 */
class NEC3Initializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        int result = NeoFacePro::Initialize();
        if (result != NFP_SUCCESS) qWarning("NEC3 Initialize error [%d]", result);
        Globals->Abbreviations.insert("NEC3", "Open+NEC3Enroll:NEC3Compare");
    }

    void finalize() const
    {
        int result = NeoFacePro::Terminate();
        if (result != NFP_SUCCESS) qWarning("NEC3 Finalize error [%d]", result);
    }
};

MM_REGISTER(Initializer, NEC3Initializer)

/*!
 * \brief Helper class
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer
 */
class CFaceInfoResourceMaker : public ResourceMaker<NeoFacePro::CFaceInfo>
{
    NeoFacePro::CFaceInfo *make() const
    {
        NeoFacePro::CFaceInfo *faceInfo = new NeoFacePro::CFaceInfo();
        faceInfo->SetParamAlgorithm(NFP_ALGORITHM003);
        faceInfo->SetParamEyesRoll(15);
        faceInfo->SetParamEyesMaxWidth(1000);
        faceInfo->SetParamEyesMinWidth(30);
        faceInfo->SetParamMaxFace(5);
        faceInfo->SetParamReliability(0);
        return faceInfo;
    }
};

/*!
 * \brief Helper class
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer
 */
class CFaceFeatureResourceMaker : public ResourceMaker<NeoFacePro::CFaceFeature>
{
    NeoFacePro::CFaceFeature *make() const
    {
        NeoFacePro::CFaceFeature *faceFeature = new NeoFacePro::CFaceFeature();
        faceFeature->SetParamFeatureType(NFP_FEATURE_S14);
        return faceFeature;
    }
};

/*!
 * \ingroup transforms
 * \brief Enroll a face image in NEC NeoFace 3
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer
 */
class NEC3Enroll : public UntrainableFeature
{
    Q_OBJECT
    Q_PROPERTY(bool detectOnly READ get_detectOnly WRITE set_detectOnly)
    MM_MEMBER(bool, detectOnly)

    Resource<NeoFacePro::CFaceInfo> faceInfoResource;
    Resource<NeoFacePro::CFaceFeature> faceFeatureResource;
    QSharedPointer<Feature> flip;

    QString parameters() const
    {
        return "bool detectOnly = 0";
    }

    void init()
    {
        faceInfoResource.setResourceMaker(new CFaceInfoResourceMaker());
        faceFeatureResource.setResourceMaker(new CFaceFeatureResourceMaker());
        faceInfoResource.setMaxResources(1); // Only works in serial
        faceFeatureResource.setMaxResources(1); // Only works in serial
        flip = QSharedPointer<Feature>(Feature::make("Flip(X)"));
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

        NeoFacePro::CFaceInfo *faceInfo = faceInfoResource.acquire();
        int result = faceInfo->FindFace(binfo, input.data);
        faceInfo->LocateEyes();
        if (result == NFP_CANNOT_FIND_FACE) {
            if (Globals->verbose) qDebug("NEC3Enroll face not found for file %s", qPrintable(src.file.flat()));
        } else if (result != NFP_SUCCESS) {
            qWarning("NEC3Enroll FindFace error %d for file %s", result, qPrintable(src.file.flat()));
        }

        NeoFacePro::CFaceFeature *faceFeature = faceFeatureResource.acquire();
        QList<cv::Point2f> landmarks;
        for (int i=0; i<faceInfo->GetFaceMax(); i++) {
            if (faceInfo->SetFaceIndex(i) != NFP_SUCCESS)
                continue;
            POINT right = faceInfo->GetRightEye();
            POINT left = faceInfo->GetLeftEye();
            landmarks.append(cv::Point2f(right.x, right.y));
            landmarks.append(cv::Point2f(left.x, left.y));

            if (detectOnly) {
                dst += src.m();
            } else {
                result = faceFeature->SetFeature(faceInfo);
                if (result != NFP_SUCCESS) {
                    qWarning("NEC3Enroll SetFeature error %d for file %s", result, qPrintable(src.file.flat()));
                    continue;
                }

                void *data;
                long size;
                result = faceFeature->Serialize(&data, &size);
                if (result != NFP_SUCCESS) {
                    qWarning("NEC3Enroll Serialize error %d for file %s", result, qPrintable(src.file.flat()));
                    continue;
                }

                dst += cv::Mat(1, size, CV_8UC1, data).clone();
                NeoFacePro::CFaceFeature::FreeSerializeData(data);
            }

            if (src.file.getBool("ForceEnrollment") && !dst.isEmpty()) break;
        }
        dst.file.appendLandmarks(landmarks);

        faceInfoResource.release(faceInfo);
        faceFeatureResource.release(faceFeature);

        if (src.file.getBool("ForceEnrollment") && dst.isEmpty()) dst += cv::Mat();
    }
};

MM_REGISTER(Feature, NEC3Enroll)

/*!
 * \ingroup distances
 * \brief Compare faces with NEC NeoFace 3 SDK
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer
 */
class NEC3Compare : public BasicComparer
{
    Q_OBJECT

    Resource<NeoFacePro::CVerifier> verifierResource;

    float compare(const Template &a, const Template &b) const
    {
        NeoFacePro::CVerifier *verifier = verifierResource.acquire();
        float score = 0;
        if (a.m().data && b.m().data) {
            int result = verifier->Verify(a.m().data, b.m().data, &score);
            if (result != NFP_SUCCESS) qWarning("NEC3Compare verify error [%d]", result);
        }
        verifierResource.release(verifier);
        return score;
    }
};

MM_REGISTER(Comparer, NEC3Compare)

#include "nec3.moc"
