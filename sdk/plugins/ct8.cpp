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
#include <mm_plugin.h>

#include "model.h"
#include "resource.h"

using namespace cv;
using namespace mm;

namespace FRsdk {
    struct OpenCVImageBody : public ImageBody
    {
        OpenCVImageBody(const Mat &m, const std::string& _name = "")
            : w(m.cols), h(m.rows), b(0), rgb(0), n(_name)
        {
            buildRgbRepresentation(m);
            buildByteRepresentation();
        }

        ~OpenCVImageBody()
        {
            delete[] rgb;
            delete[] b;
        }

        bool isColor() const { return true; }
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

        void buildByteRepresentation()
        {
            b = new Byte[w*h];
            Rgb* colorp = rgb;
            Byte* grayp = b;
            for( unsigned int i = 0; i < h; i++ ) {
                for( unsigned int k = 0; k < w; k++ ) {
                    float f = (float) colorp->r;
                    f += (float) colorp->g;
                    f += (float) colorp->b;
                    f /= 3.0f;
                    if( f > 255.0f) f = 255.0f;
                    *grayp = (Byte) f;
                    colorp++;
                    grayp++;
                }
            }
        }

    private:
        unsigned int w;
        unsigned int h;
        Byte* b;
        Rgb* rgb;
        std::string n;
    };


    struct EnrolOpenCVFeedback : public Enrollment::FeedbackBody
    {
        EnrolOpenCVFeedback(Mat *_m)
            : m(_m), firvalid(false)
        {}

        EnrolOpenCVFeedback() {}

        void start()
        {
            firvalid = false;
        }

        void processingImage(const FRsdk::Image& img) { (void) img; }
        void eyesFound( const FRsdk::Eyes::Location& eyeLoc) { (void) eyeLoc; }
        void eyesNotFound() {}
        void sampleQualityTooLow() {}
        void sampleQuality(const float& f) { (void) f; }

        void success(const FRsdk::FIR& _fir)
        {
            fir = new FRsdk::FIR(_fir);
            m->create(1, fir->size(), CV_8UC1);
            fir->serialize((Byte*)m->data);
            firvalid = true;
        }

        void failure() {}
        void end() {}

        const FRsdk::FIR& getFir() const
        {
            if (!firvalid) qFatal("FRsdk::EnrolOpenCVFeedback::getFIR no FIR.");
            return *fir;
        }

        bool firValid() const
        {
            return firvalid;
        }

    private:
        FRsdk::CountedPtr<FRsdk::FIR> fir;
        Mat *m;
        bool firvalid;
    };
}


struct CT8Initialize : public Initializer
{    
    static FRsdk::Configuration* CT8Configuration;

    void initialize() const
    {
        QFile file(":/3rdparty/ct8/activationkey.cfg");
        file.open(QFile::ReadOnly);
        QByteArray data = file.readAll();
        file.close();
        std::istringstream istream(QString(data).arg(Globals.SDKPath+"/models/ct8").toStdString());

        try {
            CT8Configuration = new FRsdk::Configuration(istream);
        } catch (std::exception &e) {
            qWarning("CT8Initialize Exception: %s", e.what());
            CT8Configuration = NULL;
        }
    }

    void finalize() const
    {
        delete CT8Configuration;
        CT8Configuration = NULL;
    }
};

FRsdk::Configuration* CT8Initialize::CT8Configuration = NULL;

MM_REGISTER(Initializer, CT8Initialize, false)


class CT8EnrollmentProcessorResource : public Resource<FRsdk::Enrollment::Processor>
{
    QSharedPointer<FRsdk::Enrollment::Processor> make() const
    {
        return QSharedPointer<FRsdk::Enrollment::Processor>(new FRsdk::Enrollment::Processor(*CT8Initialize::CT8Configuration));
    }
};


struct CT8Context
{
    CT8Context()
    {
        try {
            faceFinder = new FRsdk::Face::Finder(*CT8Initialize::CT8Configuration);
            eyesFinder = new FRsdk::Eyes::Finder(*CT8Initialize::CT8Configuration);
            firBuilder = new FRsdk::FIRBuilder(*CT8Initialize::CT8Configuration);
            facialMatchingEngine = new FRsdk::FacialMatchingEngine(*CT8Initialize::CT8Configuration);
            enrollmentProcessors.init();
        } catch (std::exception &e) {
            qFatal("CT8Context Exception: %s", e.what());
        }
    }

    ~CT8Context()
    {
        delete faceFinder;
        delete eyesFinder;
        delete firBuilder;
        delete facialMatchingEngine;
    }

    void enroll(const FRsdk::Image &img, const FRsdk::Eyes::Location &eyes, Mat *m) const
    {
        try {
            FRsdk::Sample sample(FRsdk::AnnotatedImage(img, eyes));
            FRsdk::SampleSet sampleSet;
            sampleSet.push_back(sample);

            FRsdk::Enrollment::Feedback enrollmentFeedback(new FRsdk::EnrolOpenCVFeedback(m));
            int index;
            QSharedPointer<FRsdk::Enrollment::Processor> enrollmentProcessor = enrollmentProcessors.acquire(index);
            enrollmentProcessor->process(sampleSet.begin(), sampleSet.end(), enrollmentFeedback);
            enrollmentProcessors.release(index);
        } catch (std::exception &e) {
            qFatal("CT8Context Exception: %s", e.what());
        }
    }

    static FRsdk::Position toPosition(const Point2f &point)
    {
        return FRsdk::Position(point.x, point.y);
    }

protected:
    FRsdk::Face::Finder *faceFinder;
    FRsdk::Eyes::Finder *eyesFinder;
    CT8EnrollmentProcessorResource enrollmentProcessors;
    FRsdk::FIRBuilder *firBuilder;
    FRsdk::FacialMatchingEngine *facialMatchingEngine;
};


struct CT8Detect : public UntrainableFeature
                 , public CT8Context
{
    void project(const Template &src, Template &dst) const
    {
        try {
            FRsdk::CountedPtr<FRsdk::ImageBody> i(new FRsdk::OpenCVImageBody(src));
            FRsdk::Image img(i);
            FRsdk::Face::LocationSet faceLocations = faceFinder->find(img, 0.01);

            QList<Rect> ROIs;
            QList<Point2f> landmarks;
            FRsdk::Face::LocationSet::const_iterator faceLocationSetIterator = faceLocations.begin();
            while (faceLocationSetIterator != faceLocations.end()) {
                FRsdk::Face::Location faceLocation = *faceLocationSetIterator; faceLocationSetIterator++;
                FRsdk::Eyes::LocationSet currentEyesLocations = eyesFinder->find(img, faceLocation);
                if (currentEyesLocations.size() > 0) {
                    ROIs.append(Rect(faceLocation.pos.x(), faceLocation.pos.y(), faceLocation.width, faceLocation.width));
                    landmarks.append(Point2f(currentEyesLocations.front().first.x(), currentEyesLocations.front().first.y()));
                    landmarks.append(Point2f(currentEyesLocations.front().second.x(), currentEyesLocations.front().second.y()));
                    dst += src;
                }

                if (!Globals.EnrollAll && !dst.isEmpty()) break;
            }

            dst.file.setROIs(ROIs);
            dst.file.setLandmarks(landmarks);
        } catch (std::exception &e) {
            qFatal("CT8Enroll Exception: %s", e.what());
        }

        if (!Globals.EnrollAll && dst.isEmpty()) dst += Mat();
    }
};

MM_REGISTER(Feature, CT8Detect, false)


struct CT8Enroll : public UntrainableFeature
                 , public CT8Context
{
    void project(const Template &src, Template &dst) const
    {
        try {
            FRsdk::CountedPtr<FRsdk::ImageBody> i(new FRsdk::OpenCVImageBody(src));
            FRsdk::Image img(i);

            QList<Point2f> landmarks = src.file.landmarks();
            if (landmarks.size() == 2) {
                enroll(img, FRsdk::Eyes::Location(toPosition(landmarks[0]), toPosition(landmarks[1])), dst.mp());
                dst.file.insert("CT8_First_Eye_X", landmarks[0].x);
                dst.file.insert("CT8_First_Eye_Y", landmarks[0].y);
                dst.file.insert("CT8_Second_Eye_X", landmarks[1].x);
                dst.file.insert("CT8_Second_Eye_Y", landmarks[1].y);

                QList<Rect> ROIs = src.file.ROIs();
                if (ROIs.size() == 1) {
                    dst.file.insert("CT8_Face_X", ROIs.first().x);
                    dst.file.insert("CT8_Face_Y", ROIs.first().y);
                    dst.file.insert("CT8_Face_Width", ROIs.first().width);
                    dst.file.insert("CT8_Face_Height", ROIs.first().height);
                }
            } else {
                dst = Mat();
            }
        } catch (std::exception &e) {
            qFatal("CT8Enroll Exception: %s", e.what());
        }
    }
};

MM_REGISTER(Feature, CT8Enroll, false)


struct CT8Compare : public ComparerBase,
                    public CT8Context
{
    float compare(const Mat &srcA, const Mat &srcB) const
    {
        const float DefaultNonMatchScore = 0;
        if (!srcA.data || !srcB.data) return DefaultNonMatchScore;

        float score = DefaultNonMatchScore;
        try {
            FRsdk::FIR firA = firBuilder->build((FRsdk::Byte*)srcA.data, srcA.cols);
            FRsdk::FIR firB = firBuilder->build((FRsdk::Byte*)srcB.data, srcB.cols);
            score = (float)facialMatchingEngine->compare(firA, firB);
        } catch (std::exception &e) {
            qFatal("CT8Compare Exception: %s", e.what());
        }

        return score;
    }
};

MM_REGISTER(Comparer, CT8Compare, false)
MM_REGISTER_ALGORITHM(CT8, "Open+CT8Detect!CT8Enroll:CT8Compare")
