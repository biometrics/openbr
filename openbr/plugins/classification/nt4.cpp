#include <QDebug>
#include <QFileInfo>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <NCore.h>
#include <NImages.h>
#include <NLExtractor.h>
#include <NMatcher.h>
#include <NMatcherParams.h>
#include <NTemplate.h>
#include <NLicensing.h>
#include "openbr_internal.h"

//IRIS
#include <NEExtractor.h>
#include <NEExtractorParams.h>
#include <NERecord.h>
#include <NETemplate.h>
#include <Bmp.h>
#include <NGrayscaleImage.h>

#include <openbr/core/resource.h>
#include <openbr/core/opencvutils.h>

using namespace cv;
using namespace br;

/*!
 * \ingroup initializers
 * \brief Initialize Neurotech SDK 4
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
class NT4Initializer : public Initializer
{
    Q_OBJECT

    static void manageLicenses(bool obtain)
    {
        const NChar *components = { N_T("Biometrics.FaceExtraction,Biometrics.FaceMatching,Biometrics.IrisExtraction,Biometrics.IrisMatching") };
        if (obtain) {
            NBool available;
            if (!Globals->contains("NT4_SERVER_IP")) Globals->set("NT4_SERVER_IP", "128.29.70.34");
            NResult result = NLicenseObtainComponents(N_T(qPrintable(Globals->property("NT4_SERVER_IP").toString())), N_T("5000"), components, &available);
            if (NFailed(result)) qWarning("NLicenseObtainComponents() failed, result=%i.", result);
            if (!available) qWarning("NT4 components not available.");
        } else /* release */ {
            NResult result = NLicenseReleaseComponents(components);
            if (NFailed(result)) qWarning("NLicenseReleaseComponents() failed, result=%i.", result);
        }
    }

    void initialize() const
    {
        NCoreOnStart();
        Globals->abbreviations.insert("NT4Face", "Open+NT4DetectFace!NT4EnrollFace:NT4Compare");
        Globals->abbreviations.insert("NT4Iris", "Open+NT4EnrollIris:NT4Compare");
        manageLicenses(true);
    }

    void finalize() const
    {
        manageLicenses(false);
        NCoreOnExitEx(false);
    }
};

BR_REGISTER(Initializer, NT4Initializer)

/*!
 * \brief Neurotech context
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
struct NT4Context
{
    HNLExtractor extractor; // Face extractor.
    HNEExtractor irisExtractor; // Iris extractor.
    HNMatcher matcher; // Template matcher.

    NT4Context()
    {
        NResult result;

        // Face
        // Create extractor
        result = NleCreate(&extractor);
        if (NFailed(result)) qFatal("NleCreate() failed, result=%i.", result);

        NBool detectAllFeaturePoints = true;
        NObjectSetParameter(extractor, NLEP_DETECT_ALL_FEATURE_POINTS, &detectAllFeaturePoints);

        NDouble faceConfidence = 1;
        NObjectSetParameter(extractor, NLEP_FACE_CONFIDENCE_THRESHOLD, &faceConfidence);

        NByte faceQuality = 1;
        NObjectSetParameter(extractor, NLEP_FACE_QUALITY_THRESHOLD, &faceQuality);

        NByte favorLargestFace = 0;
        NObjectSetParameter(extractor, NLEP_FAVOR_LARGEST_FACE, &favorLargestFace);

        NByte useLivenessCheck = 0;
        NObjectSetParameter(extractor, NLEP_USE_LIVENESS_CHECK, &useLivenessCheck);

        NleTemplateSize templSize = nletsLarge;
        NObjectSetParameter(extractor, NLEP_TEMPLATE_SIZE, &templSize);

        // Iris
        // Create extractor

        result = NeeCreate(&irisExtractor);
        if (NFailed(result)) qFatal("NeeCreate() failed, result=%i.", result);

        NInt boundaryPointCount = 32;
        NObjectSetParameter(irisExtractor, NEE_BOUNDARY_POINT_COUNT, &boundaryPointCount);

        NBool interlace = false;
        NObjectSetParameter(irisExtractor, NEEP_DEINTERLACE, &interlace);


        NInt innerBoundaryFrom = 40;
        NObjectSetParameter(irisExtractor, NEEP_INNER_BOUNDARY_FROM, &innerBoundaryFrom);

        NInt innerBoundaryTo = 160;
        NObjectSetParameter(irisExtractor, NEEP_INNER_BOUNDARY_TO, &innerBoundaryTo);

        NInt outerBoundaryFrom = 140;
        NObjectSetParameter(irisExtractor, NEEP_OUTER_BOUNDARY_FROM, &outerBoundaryFrom);

        NInt outerBoundaryTo = 255;
        NObjectSetParameter(irisExtractor, NEEP_OUTER_BOUNDARY_TO, &outerBoundaryTo);


        // Face
        // Create matcher
        result = NMCreate(&matcher);
        if (NFailed(result)) qFatal("NMCreate() failed, result=%i.",result);

        NInt matchingThreshold = 0;
        NObjectSetParameter(matcher, NMP_MATCHING_THRESHOLD, &matchingThreshold);
    }

    ~NT4Context()
    {
        NObjectFree(extractor);
        NObjectFree(irisExtractor);
        NObjectFree(matcher);
    }

    // to NT image
    static void toImage(const Mat &src, HNGrayscaleImage *grayscaleImage)
    {
        Mat gray;
        OpenCVUtils::cvtGray(src, gray);
        assert(gray.isContinuous());

        HNImage image;
        NResult result;
        result = NImageCreateFromDataEx(npfGrayscale, gray.cols, gray.rows, 0, gray.cols, gray.data, gray.rows*gray.cols, 0, &image);
        if (NFailed(result)) qFatal("NT4Context::toImage NImageCreateFromDataEx() failed, result=%i.", result);

        result = NImageToGrayscale(image, grayscaleImage);
        if (NFailed(result)) qFatal("NT4Context::toImage NImageToGrayscale() failed, result=%i.", result);
        NObjectFree(image);
    }

    // to OpenCV matrix
    static Mat toMat(const HNLTemplate &templ)
    {
        NSizeType bufferSize;
        NLTemplateGetSize(templ, 0, &bufferSize);

        Mat buffer(1, bufferSize, CV_8UC1);
        NLTemplateSaveToMemory(templ, buffer.data, bufferSize, 0, &bufferSize);

        return buffer;
    }

    // extract metadata
    static File toMetadata(const NleDetectionDetails &detectionDetails)
    {
        File metadata;

        metadata.insert("NT4_FaceAvailable", detectionDetails.FaceAvailable);
        metadata.insert("NT4_Face_Rectangle_X", detectionDetails.Face.Rectangle.X);
        metadata.insert("NT4_Face_Rectangle_Y", detectionDetails.Face.Rectangle.Y);
        metadata.insert("NT4_Face_Rectangle_Width", detectionDetails.Face.Rectangle.Width);
        metadata.insert("NT4_Face_Rectangle_Height", detectionDetails.Face.Rectangle.Height);
        metadata.insert("NT4_Face_Rotation_Roll", detectionDetails.Face.Rotation.Roll);
        metadata.insert("NT4_Face_Rotation_Pitch", detectionDetails.Face.Rotation.Pitch);
        metadata.insert("NT4_Face_Rotation_Yaw", detectionDetails.Face.Rotation.Yaw);
        metadata.insert("NT4_Face_Confidence", detectionDetails.Face.Confidence);

        metadata.insert("NT4_RightEyeCenter_X", detectionDetails.RightEyeCenter.X);
        metadata.insert("NT4_RightEyeCenter_Y", detectionDetails.RightEyeCenter.Y);
        metadata.insert("NT4_RightEyeCenter_Code", detectionDetails.RightEyeCenter.Code);
        metadata.insert("NT4_RightEyeCenter_Confidence", detectionDetails.RightEyeCenter.Confidence);

        metadata.insert("NT4_LeftEyeCenter_X", detectionDetails.LeftEyeCenter.X);
        metadata.insert("NT4_LeftEyeCenter_Y", detectionDetails.LeftEyeCenter.Y);
        metadata.insert("NT4_LeftEyeCenter_Code", detectionDetails.LeftEyeCenter.Code);
        metadata.insert("NT4_LeftEyeCenter_Confidence", detectionDetails.LeftEyeCenter.Confidence);

        metadata.insert("NT4_MouthCenter_X", detectionDetails.MouthCenter.X);
        metadata.insert("NT4_MouthCenter_Y", detectionDetails.MouthCenter.Y);
        metadata.insert("NT4_MouthCenter_Code", detectionDetails.MouthCenter.Code);
        metadata.insert("NT4_MouthCenter_Confidence", detectionDetails.MouthCenter.Confidence);

        metadata.insert("NT4_NoseTip_X", detectionDetails.NoseTip.X);
        metadata.insert("NT4_NoseTip_Y", detectionDetails.NoseTip.Y);
        metadata.insert("NT4_NoseTip_Code", detectionDetails.NoseTip.Code);
        metadata.insert("NT4_NoseTip_Confidence", detectionDetails.NoseTip.Confidence);

        return metadata;
    }

    // Initialize from metadata
    static NleDetectionDetails fromMetadata(const File &metadata)
    {
        NleDetectionDetails detectionDetails;

        detectionDetails.FaceAvailable = metadata.value("NT4_FaceAvailable").toBool();
        detectionDetails.Face.Rectangle.X = metadata.value("NT4_Face_Rectangle_X").toInt();
        detectionDetails.Face.Rectangle.Y = metadata.value("NT4_Face_Rectangle_Y").toInt();
        detectionDetails.Face.Rectangle.Width = metadata.value("NT4_Face_Rectangle_Width").toInt();
        detectionDetails.Face.Rectangle.Height = metadata.value("NT4_Face_Rectangle_Height").toInt();
        detectionDetails.Face.Rotation.Roll = metadata.value("NT4_Face_Rotation_Roll").toInt();
        detectionDetails.Face.Rotation.Pitch = metadata.value("NT4_Face_Rotation_Pitch").toInt();
        detectionDetails.Face.Rotation.Yaw = metadata.value("NT4_Face_Rotation_Yaw").toInt();
        detectionDetails.Face.Confidence = metadata.value("NT4_Face_Confidence").toDouble();

        detectionDetails.RightEyeCenter.X = metadata.value("NT4_RightEyeCenter_X").toInt();
        detectionDetails.RightEyeCenter.Y = metadata.value("NT4_RightEyeCenter_Y").toInt();
        detectionDetails.RightEyeCenter.Code = metadata.value("NT4_RightEyeCenter_Code").toInt();
        detectionDetails.RightEyeCenter.Confidence = metadata.value("NT4_RightEyeCenter_Confidence").toDouble();

        detectionDetails.LeftEyeCenter.X = metadata.value("NT4_LeftEyeCenter_X").toInt();
        detectionDetails.LeftEyeCenter.Y = metadata.value("NT4_LeftEyeCenter_Y").toInt();
        detectionDetails.LeftEyeCenter.Code = metadata.value("NT4_LeftEyeCenter_Code").toInt();
        detectionDetails.LeftEyeCenter.Confidence = metadata.value("NT4_LeftEyeCenter_Confidence").toDouble();

        detectionDetails.MouthCenter.X = metadata.value("NT4_MouthCenter_X").toInt();
        detectionDetails.MouthCenter.Y = metadata.value("NT4_MouthCenter_Y").toInt();
        detectionDetails.MouthCenter.Code = metadata.value("NT4_MouthCenter_Code").toInt();
        detectionDetails.MouthCenter.Confidence = metadata.value("NT4_MouthCenter_Confidence").toDouble();

        detectionDetails.NoseTip.X = metadata.value("NT4_NoseTip_X").toInt();
        detectionDetails.NoseTip.Y = metadata.value("NT4_NoseTip_Y").toInt();
        detectionDetails.NoseTip.Code = metadata.value("NT4_NoseTip_Code").toInt();
        detectionDetails.NoseTip.Confidence = metadata.value("NT4_NoseTip_Confidence").toDouble();

        return detectionDetails;
    }
};

/*!
 * \ingroup transforms
 * \brief Neurotech face detection
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
class NT4DetectFace : public UntrainableTransform
{
    Q_OBJECT

    Resource<NT4Context> contexts;

public:
    NT4DetectFace() : UntrainableTransform(true) {}

private:
    void project(const Template &src, Template &dst) const
    {
        HNGrayscaleImage grayscaleImage;
        NT4Context::toImage(src, &grayscaleImage);

        NT4Context *context = contexts.acquire();

        NInt faceCount;
        NleFace *faces;
        NResult result = NleDetectFaces(context->extractor, grayscaleImage, &faceCount, &faces);
        if (NFailed(result)) qFatal("NT4DetectFace::project NleDetectFaces() failed, result=%i.", result);
        for (int i=0; i<faceCount; i++) {
            NleDetectionDetails detectionDetails;
            result = NleDetectFacialFeatures(context->extractor, grayscaleImage, &faces[i], &detectionDetails);
            if (NFailed(result)) qFatal("NT4DetectFace::project NleDetectFacialFeatures() failed, result=%i.", result);

            dst.file.append(NT4Context::toMetadata(detectionDetails));
            dst += src;
            //if (!Globals.EnrollAll) break;
        }

        contexts.release(context);
        NObjectFree(grayscaleImage);

        //if (!Globals.EnrollAll && dst.isEmpty()) dst = Mat();
        if (dst.isEmpty()) dst = Mat();
    }
};

BR_REGISTER(Transform, NT4DetectFace)

/*!
 * \ingroup transforms
 * \brief Enroll face in Neurotech SDK 4
 * \author Josh Klontz \cite jklontz
 */
class NT4EnrollFace : public UntrainableTransform
{
    Q_OBJECT

    Resource<NT4Context> contexts;

public:
    NT4EnrollFace() : UntrainableTransform(true) {}

private:
    void project(const Template &src, Template &dst) const
    {
        if (!src.m().data) {
            dst = Mat();
            return;
        }

        HNGrayscaleImage grayscaleImage;
        NT4Context::toImage(src, &grayscaleImage);

        NT4Context *context = contexts.acquire();

        NleDetectionDetails detectionDetails = NT4Context::fromMetadata(src.file);
        NleExtractionStatus extractionStatus;
        HNLTemplate templ;

        NResult result = NleExtract(context->extractor, grayscaleImage, &detectionDetails, &extractionStatus, &templ);
        contexts.release(context);

        if (NFailed(result) || (extractionStatus != nleesTemplateCreated))
            dst = Mat();
        else
            dst = NT4Context::toMat(templ);

        NObjectFree(templ);
        NObjectFree(grayscaleImage);
    }
};

BR_REGISTER(Transform, NT4EnrollFace)

/*!
 * \ingroup transforms
 * \brief Enroll iris in Neurotech SDK 4
 * \author E. Taborsky \cite mmtaborsky
 */
class NT4EnrollIris : public UntrainableTransform
{
    Q_OBJECT

    Resource<NT4Context> contexts;

public:
    NT4EnrollIris() : UntrainableTransform(true) {}

private:
    void project(const Template &src, Template &dst) const
    {
        HNGrayscaleImage grayscaleImage;
        NT4Context::toImage(src, &grayscaleImage);

        NeeSegmentationDetails segmentationDetails;
        NeeExtractionStatus extractionStatus;
        HNERecord hRecord;

        NResult result = NERecordCreate((NUShort)src.m().cols,(NUShort)src.m().rows, 0, &hRecord); // This seems wrong...
        assert(!NFailed(result));

        NT4Context *context = contexts.acquire();


        result = NeeExtract(context->irisExtractor, grayscaleImage, nepUnknown, &segmentationDetails, &extractionStatus, &hRecord);

        if (!(segmentationDetails.OuterBoundaryAvailable)){
            qDebug("NT4EnrollIris::project Outer Boundary not available");
        }

        if (NFailed(result)) qFatal("NT4EnrollIris::project NeeExtract() failed, result=%i.", result);
        else if (extractionStatus == neeesTemplateCreated){
            NSizeType bufferSize;
            NERecordGetSize(hRecord, 0, &bufferSize);

            Mat buffer(1, bufferSize, CV_8UC1);
            NERecordSaveToMemory(hRecord, buffer.data, bufferSize, 0, &bufferSize);

            dst = Template(src.file, buffer);
        }

        contexts.release(context);
        NObjectFree(grayscaleImage);

        //if (!Globals.EnrollAll && dst.isEmpty()) dst.append(Template(src.file, Mat()));
        if (dst.isEmpty()) dst.append(Template(src.file, Mat()));
    }
};

BR_REGISTER(Transform, NT4EnrollIris)

/*!
 * \ingroup distances
 * \brief Compare templates with Neurotech SDK 4
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
class NT4Compare : public Distance
{
    Q_OBJECT

    Resource<NT4Context> contexts;

    float compare(const br::Template &a, const br::Template &b) const
    {
        NT4Context *context = contexts.acquire();

        NResult result;

        const Mat &srcA = a;
        if (srcA.data) {
            result = NMIdentifyStartEx(context->matcher, srcA.data, srcA.rows*srcA.cols, NULL);
            if (NFailed(result)) qFatal("NT4Compare::compare NMIdentifyStart() failed, result=%i.", result);
        }

        const Mat &srcB = b;
        float score = -std::numeric_limits<float>::max();
        if (srcA.data && srcB.data) {
            NInt pScore;
            result = NMIdentifyNextEx(context->matcher, srcB.data, srcB.rows*srcB.cols, NULL, &pScore);
            if (NFailed(result)) qFatal("NT4Compare::compare NMIdentifyNext() failed, result=%i.",result);
            score = float(pScore);
        }

        if (srcA.data) {
            result = NMIdentifyEnd(context->matcher);
            if (NFailed(result)) qFatal("NT4Compare::compare NMIdentifyEnd() failed, result=%i.", result);
        }

        contexts.release(context);
        return score;
    }
};

BR_REGISTER(Distance, NT4Compare)

#include "classification/nt4.moc"
