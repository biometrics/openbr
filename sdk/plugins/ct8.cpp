#include <QMap>
#include <QVariant>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <frsdk/config.h>
#include <frsdk/cptr.h>
#include <frsdk/enroll.h>
#include <frsdk/face.h>
#include <frsdk/image.h>
#include <frsdk/match.h>
#include <exception>
#include <string>
#include <vector>
#include <openbr_plugin.h>

#include "core/resource.h"

using namespace cv;
using namespace br;

#define CT8_CONFIG_PROP "ct8ConfigFile"

namespace FRsdk {
    // Construct a FaceVACS sdk ImageBody from an opencv Mat
    struct OpenCVImageBody : public ImageBody
    {
        OpenCVImageBody(const Mat &m, const std::string& _name = "")
            : w(m.cols), h(m.rows), b(0), rgb(0), n(_name)
        {
            // The ImageBody only needs to construct a grayscale or rgb
            // representation (whichever is indicated by isColor()), not both.
            if (m.channels() == 1) {
                is_color = false;
                buildByteRepresentation(m);
            }
            else {
                buildRgbRepresentation(m);
                is_color = true;
            }
            
        }

        ~OpenCVImageBody()
        {
            delete[] rgb;
            delete[] b;
        }

        bool isColor() const { return is_color; }
        unsigned int height() const { return h; }
        unsigned int width() const { return w; }
        const Byte* grayScaleRepresentation() const { return b; }
        const Rgb* colorRepresentation() const { return rgb; }
        std::string name() const { return n;}

        void buildRgbRepresentation(const Mat &m)
        {
            // layout required by FRsdk::Images: [Blue Green Red NotUsed]; no padding
            rgb = new Rgb[w * h * sizeof(Rgb)];
            std::vector<Mat> mv;
            split(m, mv);
            for (unsigned int i = 0; i < h; i++) {
                for (unsigned int k = 0; k < w; k++) {
                    rgb[i*w+k].b = mv[0 % mv.size()].at<uchar>(i, k);
                    rgb[i*w+k].g = mv[1 % mv.size()].at<uchar>(i, k);
                    rgb[i*w+k].r = mv[2 % mv.size()].at<uchar>(i, k);
                    rgb[i*w+k].a = 0;
                }
            }
        }

        void buildByteRepresentation(const Mat & m)
        {
            b = new Byte[w*h];
            Byte* grayp = b;
            for( unsigned int i = 0; i < h; i++ ) {
                for( unsigned int k = 0; k < w; k++ ) {
                    *grayp = (Byte) m.at<uchar>(i,k);
                    grayp++;
                }
            }
        }

    private:
        bool is_color;
        unsigned int w;
        unsigned int h;
        Byte* b;
        Rgb* rgb;
        std::string n;
    };


    // Enrollment::FeedbackBody subclasses are used as a set of callbacks
    // during facevacs enrollment. This class keeps track of whether or not
    // enrollment has failed (checkable via firValid()), and the extracted fir
    // (getFir)
    struct EnrolOpenCVFeedback : public Enrollment::FeedbackBody
    {
        EnrolOpenCVFeedback(Mat *_m)
            : m(_m), firvalid(false)
        {}

        EnrolOpenCVFeedback() {}

        void start() { firvalid = false; }

        void processingImage(const FRsdk::Image& img) { (void) img; }
        void eyesFound( const FRsdk::Eyes::Location& eyeLoc) { (void) eyeLoc; }
        void eyesNotFound() { firvalid = false;}
        void sampleQualityTooLow() {}
        void sampleQuality(const float& f) { (void) f; }

        void success(const FRsdk::FIR& _fir)
        {
            fir = new FRsdk::FIR(_fir);
            m->create(1, fir->size(), CV_8UC1);
            fir->serialize((Byte*)m->data);
            firvalid = true;
        }

        void failure() { firvalid = false; }
        void end() {}

        const FRsdk::FIR& getFir() const
        {
            if (!firvalid) qFatal("FRsdk::EnrolOpenCVFeedback::getFIR no FIR.");
            return *fir;
        }

        bool firValid() const { return firvalid; }

    private:
        FRsdk::CountedPtr<FRsdk::FIR> fir;
        Mat *m;
        bool firvalid;
    };
}

/*!
 * \ingroup initializers
 * \brief Initialize ct8 plugin
 * \author Josh Klontz \cite jklontz
 * \author Charles Otto \cite caotto
 */
struct CT8Initialize : public Initializer
{    
    static FRsdk::Configuration* CT8Configuration;
public:
    // ct8 plugin initialization, load a FRsdk config file, and register
    // the shortcut for using FaceVACS feature extraction/comparison
    void initialize() const
    {
        try {
            QString store_string = QString((CT8_DIR + std::string("/etc/frsdk.cfg")).c_str());
            Globals->setProperty(CT8_CONFIG_PROP, store_string);
            CT8Configuration = NULL;
            Globals->abbreviations.insert("CT8","Open+CT8Detect!CT8Enroll:CT8Compare");
        } catch (std::exception &e) {
            qWarning("CT8Initialize Exception: %s", e.what());
            CT8Configuration = NULL;
        }
    }

    static FRsdk::Configuration * getCT8Configuration()
    {
        if (!CT8Configuration) {
            QVariant recovered_variant= Globals->property(CT8_CONFIG_PROP);
            QString recovered_string = recovered_variant.toString();
            try {
                CT8Configuration = new FRsdk::Configuration(qPrintable(recovered_string));
            } catch (std::exception &e) {
                qFatal("CT8Initialize Exception: %s", e.what());
            }
        }
        return CT8Configuration;
    }


    void finalize() const
    {
        delete CT8Configuration;
        CT8Configuration = NULL;
    }
};

FRsdk::Configuration* CT8Initialize::CT8Configuration = NULL;

BR_REGISTER(Initializer, CT8Initialize)

// Adaptor class adding a default constructor to FRsdk::Enrollment::Processor
// so that it can be used with Resource
class CT8EnrollmentProcessor : public FRsdk::Enrollment::Processor
{
public:
    CT8EnrollmentProcessor() : FRsdk::Enrollment::Processor(*CT8Initialize::getCT8Configuration())
    {
        //
    }
};

typedef Resource<CT8EnrollmentProcessor> CT8EnrollmentProcessorResource;

/*!
 * \brief CT8 context
 * \author Josh Klontz \cite jklontz
 * \author Charles Otto \cite caotto
 */
struct CT8Context
{
    CT8Context()
    {
        try {
            faceFinder = new FRsdk::Face::Finder(*CT8Initialize::getCT8Configuration());
            eyesFinder = new FRsdk::Eyes::Finder(*CT8Initialize::getCT8Configuration());
            firBuilder = new FRsdk::FIRBuilder(*CT8Initialize::getCT8Configuration());
            facialMatchingEngine = new FRsdk::FacialMatchingEngine(*CT8Initialize::getCT8Configuration());
        } catch (std::exception &e) {
            qFatal("CT8Context Exception: %s", e.what());
        }
    }

    virtual ~CT8Context()
    {
        delete faceFinder;
        delete eyesFinder;
        delete firBuilder;
        delete facialMatchingEngine;
    }

    // Enroll an FRsdk::Sample (can be various types, generally an image that
    // maybe has some extra data like detected eye locations). 
    bool enroll(const FRsdk::Sample &sample, Mat *m) const
    {
        try {
            FRsdk::SampleSet sampleSet;
            sampleSet.push_back(sample);


            FRsdk::EnrolOpenCVFeedback * feedback_body = new FRsdk::EnrolOpenCVFeedback(m);
            FRsdk::CountedPtr<FRsdk::Enrollment::FeedbackBody> feedback_ptr(feedback_body);
            
            FRsdk::Enrollment::Feedback enrollmentFeedback(feedback_ptr);

            CT8EnrollmentProcessor * enrollmentProcessor = enrollmentProcessors.acquire();
            enrollmentProcessor->process(sampleSet.begin(), sampleSet.end(), enrollmentFeedback);
            enrollmentProcessors.release(enrollmentProcessor);
            if (!feedback_body->firValid()) return false;
        } catch (std::exception &e) {
            qFatal("CT8Context Exception: %s", e.what());
            return false;
        }
        return true;
    }


    // Input: an image, and pre-detected eye locations, returns false if enrollment fails
    bool enroll(const FRsdk::Image &img, const FRsdk::Eyes::Location &eyes, Mat *m) const
    {
        try {
            FRsdk::Sample sample(FRsdk::AnnotatedImage(img, eyes));
            return enroll(sample, m);
        } catch (std::exception &e) {
            qFatal("CT8Context Exception: %s", e.what());
            return false;
        }
        return true;
    }

    // Input: an image, no eye locations (facevacs will do detection with
    // default parameters. Returns false if enrollment fails
    bool enroll(const FRsdk::Image &img, Mat *m) const
    {
        try {
            FRsdk::Sample sample(img);
            return enroll(sample, m);
        } catch (std::exception &e) {
            qFatal("CT8Context Exception: %s", e.what());
            return false;
        }
        return true;
    }

	static FRsdk::Position toPosition(const QPointF &point)
    {
        return FRsdk::Position(point.x(), point.y());
    }

protected:
    FRsdk::Face::Finder *faceFinder;
    FRsdk::Eyes::Finder *eyesFinder;
    CT8EnrollmentProcessorResource enrollmentProcessors;
    FRsdk::FIRBuilder *firBuilder;
    FRsdk::FacialMatchingEngine *facialMatchingEngine;
};

/*!
 * \ingroup transforms
 * \brief Perform face and eye detection with the FaceVACS SDK.
 * \author Josh Klontz \cite jklontz
 * \author Charles Otto \cite caotto
 */
struct CT8Detect : public UntrainableTransform
                 , public CT8Context
{
public:
    Q_OBJECT

    Q_PROPERTY(float minRelEyeDistance READ get_minRelEyeDistance WRITE set_minRelEyeDistance RESET reset_minRelEyeDistance STORED false)
    Q_PROPERTY(float maxRelEyeDistance READ get_maxRelEyeDistance WRITE set_maxRelEyeDistance RESET reset_maxRelEyeDistance STORED false)

    // Perform face, then eye detection using the facevacs SDK
    void project(const Template &src, Template &dst) const
    {
        try {
            // Build an FRsdk image from the input openCV mat
            FRsdk::CountedPtr<FRsdk::ImageBody> i(new FRsdk::OpenCVImageBody(src));
            FRsdk::Image img(i);
            
            FRsdk::Face::LocationSet faceLocations = faceFinder->find(img, minRelEyeDistance, maxRelEyeDistance);
            
            // If the face finder doesn't find anything mark the output as a failure
            if (faceLocations.empty() ) {
                dst.file.setBool("FTE");
                return;
            }

            QList<QRectF> ROIs;
            QList<QPointF> landmarks;
            FRsdk::Face::LocationSet::const_iterator faceLocationSetIterator = faceLocations.begin();
            bool any_eyes = false;

            // Attempt to detect eyes in any face ROIs that were detected
            while (faceLocationSetIterator != faceLocations.end()) {
                FRsdk::Face::Location faceLocation = *faceLocationSetIterator; faceLocationSetIterator++;
                FRsdk::Eyes::LocationSet currentEyesLocations = eyesFinder->find(img, faceLocation);

                if (currentEyesLocations.size() > 0) {
                    any_eyes = true;
                    ROIs.append(QRectF(faceLocation.pos.x(), faceLocation.pos.y(), faceLocation.width, faceLocation.width));
                    landmarks.append(QPointF(currentEyesLocations.front().first.x(), currentEyesLocations.front().first.y()));
                    landmarks.append(QPointF(currentEyesLocations.front().second.x(), currentEyesLocations.front().second.y()));

                    dst += src;
                }

                if (any_eyes && !Globals->enrollAll && !dst.isEmpty()) break;
            }

            // If eye detection failed, mark the output as a failure
            if (!any_eyes) {
                dst.file.setBool("FTE");
                return;
            }

            dst.file.setROIs(ROIs);
            dst.file.setLandmarks(landmarks);
        } catch (std::exception &e) {
            qFatal("CT8Enroll Exception: %s", e.what());
        }

        if (!Globals->enrollAll && dst.isEmpty()) dst += Mat();
    }
private:
    BR_PROPERTY(float, minRelEyeDistance, 0.01f)
    BR_PROPERTY(float, maxRelEyeDistance, 0.4f)
};

BR_REGISTER(Transform, CT8Detect)

/*!
 * \ingroup transforms
 * \brief Enroll face images using the FaceVACS SDK
 * \author Josh Klontz \cite jklontz
 * \author Charles Otto \cite caotto
 */
struct CT8Enroll : public UntrainableTransform
                 , public CT8Context
{
    Q_OBJECT
    // enroll an image using the facevacs sdk. Generates a facevacs "fir" which
    // is their face representation.
    void project(const Template &src, Template &dst) const
    {
        try {
            FRsdk::CountedPtr<FRsdk::ImageBody> i(new FRsdk::OpenCVImageBody(src));
            FRsdk::Image img(i);

            // If we already have eye locations, use them 
            QList<QPointF> landmarks = src.file.landmarks();
            bool enroll_succeeded = false;
            if (landmarks.size() == 2) {
                enroll_succeeded = enroll(img, FRsdk::Eyes::Location(toPosition(landmarks[0]), toPosition(landmarks[1])), &(dst.m()));

                // Transfer previously detectd eye and face locations to the output dst.
                dst.file.insert("CT8_First_Eye_X", landmarks[0].x());
                dst.file.insert("CT8_First_Eye_Y", landmarks[0].y());
                dst.file.insert("CT8_Second_Eye_X", landmarks[1].x());
                dst.file.insert("CT8_Second_Eye_Y", landmarks[1].y());

                QList<QRectF> ROIs = src.file.ROIs();
                if (ROIs.size() == 1) {
                    dst.file.insert("CT8_Face_X", ROIs.first().x());
                    dst.file.insert("CT8_Face_Y", ROIs.first().y());
                    dst.file.insert("CT8_Face_Width", ROIs.first().width());
                    dst.file.insert("CT8_Face_Height", ROIs.first().height());
                }
            } else {
                // If we don't have eye locations already, calling enroll here
                // will cause facevacs to perform detection using default
                // parameters (and we will not receive the detected locations
                // as output).
                enroll_succeeded = enroll(img, &(dst.m()));
            }
            // If enrollment failed, mark this image as a failure. This will
            // typically only happen if we aren't using pre-detected eye
            // locations
            if (!enroll_succeeded)
            {
                dst.file.setBool("FTE");
                return;
            }

        } catch (std::exception &e) {
            qFatal("CT8Enroll Exception: %s", e.what());
        }
    }
};

BR_REGISTER(Transform, CT8Enroll)

/*!
 * \ingroup distances
 * \brief Compare FaceVACS templates 
 * \author Josh Klontz \cite jklontz
 * \author Charles Otto \cite caotto
 */
struct CT8Compare : public Distance,
                    public CT8Context
{
    Q_OBJECT

    // Compare pre-extracted facevacs templates
    float compare(const Template &srcA, const Template &srcB) const
    {
        const float DefaultNonMatchScore = 0;
        if (!srcA.m().data || !srcB.m().data) return DefaultNonMatchScore;

        float score = DefaultNonMatchScore;
        try {
            FRsdk::FIR firA = firBuilder->build( (FRsdk::Byte *) srcA.m().data, srcA.m().cols);
            FRsdk::FIR firB = firBuilder->build( (FRsdk::Byte *) srcB.m().data, srcB.m().cols);
            score = (float)facialMatchingEngine->compare(firA, firB);
        } catch (std::exception &e) {
            qFatal("CT8Compare Exception: %s", e.what());
        }

        return score;
    }
};

BR_REGISTER(Distance, CT8Compare)

#include "plugins/ct8.moc"

