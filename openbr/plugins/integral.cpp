#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <openbr/openbr_plugin.h>

#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Computes integral image.
 * \author Josh Klontz \cite jklontz
 */
class IntegralTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        integral(src, dst);
    }
};

BR_REGISTER(Transform, IntegralTransform)

/*!
 * \ingroup transforms
 * \brief Sliding window feature extraction from a multi-channel integral image.
 * \author Josh Klontz \cite jklontz
 */
class IntegralSamplerTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int scales READ get_scales WRITE set_scales RESET reset_scales STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float stepFactor READ get_stepFactor WRITE set_stepFactor RESET reset_stepFactor STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(bool secondOrder READ get_secondOrder WRITE set_secondOrder RESET reset_secondOrder STORED false)
    BR_PROPERTY(int, scales, 6)
    BR_PROPERTY(float, scaleFactor, 2)
    BR_PROPERTY(float, stepFactor, 0.75)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(bool, secondOrder, false)

    void project(const Template &src, Template &dst) const
    {
        typedef Eigen::Map< const Eigen::Matrix<qint32,Eigen::Dynamic,1> > InputDescriptor;
        typedef Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,1> > SecondOrderInputDescriptor;
        typedef Eigen::Map< Eigen::Matrix<float,Eigen::Dynamic,1> > OutputDescriptor;

        const Mat &m = src.m();
        if (m.depth() != CV_32S) qFatal("Expected CV_32S matrix depth.");
        const int channels = m.channels();
        const int rowStep = channels * m.cols;

        int descriptors = 0;
        float idealSize = min(m.rows, m.cols)-1;
        for (int scale=0; scale<scales; scale++) {
            const int currentSize(idealSize);
            const int numDown = 1+(m.rows-currentSize-1)/int(idealSize*stepFactor);
            const int numAcross = 1+(m.cols-currentSize-1)/int(idealSize*stepFactor);
            descriptors += numDown*numAcross;
            if (secondOrder) descriptors += numDown*(numAcross-1) + (numDown-1)*numAcross;
            idealSize /= scaleFactor;
            if (idealSize < minSize) break;
        }
        Mat n(descriptors, channels, CV_32FC1);

        const qint32 *dataIn = (qint32*)m.data;
        float *dataOut = (float*)n.data;
        idealSize = min(m.rows, m.cols)-1;
        int index = 0;
        for (int scale=0; scale<scales; scale++) {
            const int currentSize(idealSize);
            const int currentStep(idealSize*stepFactor);
            for (int i=currentSize; i<m.rows; i+=currentStep) {
                for (int j=currentSize; j<m.cols; j+=currentStep) {
                    InputDescriptor a(dataIn+((i-currentSize)*rowStep+(j-currentSize)*channels), channels, 1);
                    InputDescriptor b(dataIn+((i-currentSize)*rowStep+ j             *channels), channels, 1);
                    InputDescriptor c(dataIn+(i              *rowStep+(j-currentSize)*channels), channels, 1);
                    InputDescriptor d(dataIn+(i              *rowStep+ j             *channels), channels, 1);
                    OutputDescriptor y(dataOut+(index*channels), channels, 1);
                    y = (d-b-c+a).cast<float>()/(currentSize*currentSize);
                    index++;
                }
            }
            if (secondOrder) {
                const int numDown = 1+(m.rows-currentSize-1)/currentStep;
                const int numAcross = 1+(m.cols-currentSize-1)/currentStep;
                const float *dataIn = n.ptr<float>(index - numDown*numAcross);
                for (int i=0; i<numDown; i++) {
                    for (int j=0; j<numAcross; j++) {
                        SecondOrderInputDescriptor a(dataIn + (i*numAcross+j)*channels, channels, 1);
                        if (j < numAcross-1) {
                            OutputDescriptor y(dataOut+(index*channels), channels, 1);
                            y = a - SecondOrderInputDescriptor(dataIn + (i*numAcross+j+1)*channels, channels, 1);
                            index++;
                        }
                        if (i < numDown-1) {
                            OutputDescriptor y(dataOut+(index*channels), channels, 1);
                            y = a - SecondOrderInputDescriptor(dataIn + ((i+1)*numAcross+j)*channels, channels, 1);
                            index++;
                        }
                    }
                }
            }
            idealSize /= scaleFactor;
            if (idealSize < minSize) break;
        }

        if (descriptors != index)
            qFatal("Allocated %d descriptors but computed %d.", descriptors, index);

        dst.m() = n;
    }
};

BR_REGISTER(Transform, IntegralSamplerTransform)

/*!
 * \ingroup transforms
 * \brief Construct template in a recursive decent manner.
 * \author Josh Klontz \cite jklontz
 */
class RecursiveIntegralSamplerTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int scales READ get_scales WRITE set_scales RESET reset_scales STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(int, scales, 6)
    BR_PROPERTY(float, scaleFactor, 2)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(br::Transform*, transform, NULL)

    Transform *subTransform;

    typedef Eigen::Map< const Eigen::Matrix<qint32,Eigen::Dynamic,1> > InputDescriptor;
    typedef Eigen::Map< Eigen::Matrix<float,Eigen::Dynamic,1> > OutputDescriptor;
    typedef Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,1> > SecondOrderInputDescriptor;

    void init()
    {
        if (scales >= 2) {
            File subFile = file;
            subFile.set("scales", scales-1);
            subTransform = make(subFile.flat());
        } else {
            subTransform = NULL;
        }
    }

    static void integralHistogram(const Mat &src, const int x, const int y, const int rows, const int columns, Mat &dst, int index)
    {
        const int channels = src.channels();
        OutputDescriptor(dst.ptr<float>(index), channels, 1) =
            (  InputDescriptor(src.ptr<qint32>(y+rows, x+columns), channels, 1)
             - InputDescriptor(src.ptr<qint32>(y,      x+columns), channels, 1)
             - InputDescriptor(src.ptr<qint32>(y+rows, x),         channels, 1)
             + InputDescriptor(src.ptr<qint32>(y,      x),         channels, 1)).cast<float>()/(rows*columns);
    }

    void computeDescriptor(const Mat &src, Mat &dst) const
    {
        const int channels = src.channels();
        const int rows = src.rows-1; // Integral images have an extra row and column
        const int columns = src.cols-1;

        Mat tmp(5, channels, CV_32FC1);
        integralHistogram(src, 0,      0,         rows/2, columns/2, tmp, 0);
        integralHistogram(src, 0,      columns/2, rows/2, columns/2, tmp, 1);
        integralHistogram(src, rows/2, 0,         rows/2, columns/2, tmp, 2);
        integralHistogram(src, rows/2, columns/2, rows/2, columns/2, tmp, 3);
        integralHistogram(src, rows/4, columns/4, rows/2, columns/2, tmp, 4);
        const SecondOrderInputDescriptor a(tmp.ptr<float>(0), channels, 1);
        const SecondOrderInputDescriptor b(tmp.ptr<float>(1), channels, 1);
        const SecondOrderInputDescriptor c(tmp.ptr<float>(2), channels, 1);
        const SecondOrderInputDescriptor d(tmp.ptr<float>(3), channels, 1);
        const SecondOrderInputDescriptor e(tmp.ptr<float>(4), channels, 1);

        dst = Mat(5, channels, CV_32FC1);
        OutputDescriptor(dst.ptr<float>(0), channels, 1) = (a+b+c+d)/4.f;
        OutputDescriptor(dst.ptr<float>(1), channels, 1) = ((a+b+c+d)/4.f-e);
        OutputDescriptor(dst.ptr<float>(2), channels, 1) = ((a+b)-(c+d))/2.f;
        OutputDescriptor(dst.ptr<float>(3), channels, 1) = ((a+c)-(b+d))/2.f;
        OutputDescriptor(dst.ptr<float>(4), channels, 1) = ((a+d)-(b+c))/2.f;
    }

    Template subdivide(const Template &src) const
    {
        // Integral images have an extra row and column
        int subWidth = (src.m().cols-1) / scaleFactor + 1;
        int subHeight = (src.m().rows-1) / scaleFactor + 1;
        return Template(src.file, QList<Mat>() << Mat(src, Rect(0,                     0, subWidth, subHeight))
                                               << Mat(src, Rect(src.m().cols-subWidth, 0, subWidth, subHeight))
                                               << Mat(src, Rect(0,                     src.m().rows-subHeight, subWidth, subHeight))
                                               << Mat(src, Rect(src.m().cols-subWidth, src.m().rows-subHeight, subWidth, subHeight)));
    }

    bool canSubdivide(const Template &t) const
    {
        // Integral images have an extra row and column
        const int subWidth = (t.m().cols-1) / scaleFactor;
        const int subHeight = (t.m().rows-1) / scaleFactor;
        return ((subWidth >= minSize) && (subHeight >= minSize));
    }

    void train(const TemplateList &src)
    {
        if (subTransform != NULL) {
            TemplateList subSrc; subSrc.reserve(src.size());
            foreach (const Template &t, src)
                if (canSubdivide(t))
                    subSrc.append(subdivide(t));

            if (subSrc.isEmpty()) {
                delete subTransform;
                subTransform = NULL;
            } else {
                subTransform->train(subSrc);
            }
        }

        TemplateList dst; dst.reserve(src.size());
        foreach (const Template &t, src) {
            Template u(t.file);
            computeDescriptor(t, u);
            dst.append(u);
        }
        transform->train(dst);
    }

    void project(const Template &src, Template &dst) const
    {
        computeDescriptor(src, dst);
        transform->project(dst, dst);

        if ((subTransform != NULL) && canSubdivide(src)) {
            Template subDst;
            subTransform->project(subdivide(src), subDst);
            dst.append(subDst);
        }
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
        stream << (subTransform != NULL);
        if (subTransform != NULL)
            subTransform->store(stream);
    }

    void load(QDataStream &stream)
    {
        transform->load(stream);
        bool hasSubTransform;
        stream >> hasSubTransform;
        if (hasSubTransform) subTransform->load(stream);
        else                 { delete subTransform; subTransform = NULL; }
    }
};

BR_REGISTER(Transform, RecursiveIntegralSamplerTransform)

/*!
 * \ingroup transforms
 * \brief Computes magnitude and/or angle of image.
 * \author Josh Klontz \cite jklontz
 */
class GradientTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Channel)
    Q_PROPERTY(Channel channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    enum Channel { Magnitude, Angle, MagnitudeAndAngle };

private:
    BR_PROPERTY(Channel, channel, Angle)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC1) qFatal("Requires CV_8UC1 input.");
        Mat dx, dy, magnitude, angle;
        Sobel(src, dx, CV_32F, 1, 0);
        Sobel(src, dy, CV_32F, 0, 1);
        cartToPolar(dx, dy, magnitude, angle, true);
        if ((channel == Magnitude) || (channel == MagnitudeAndAngle))
            dst.append(magnitude);
        if ((channel == Angle) || (channel == MagnitudeAndAngle))
            dst.append(angle);
    }
};

BR_REGISTER(Transform, GradientTransform)

/*!
 * \ingroup transforms
 * \brief Projects each row based on a computed word.
 * \author Josh Klontz \cite jklontz
 */
class WordWiseTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* getWords READ get_getWords WRITE set_getWords RESET reset_getWords)
    Q_PROPERTY(br::Transform* byWord READ get_byWord WRITE set_byWord RESET reset_byWord)
    Q_PROPERTY(int numWords READ get_numWords WRITE set_numWords RESET reset_numWords)
    BR_PROPERTY(br::Transform*, getWords, NULL)
    BR_PROPERTY(br::Transform*, byWord, NULL)
    BR_PROPERTY(int, numWords, 0)

    void train(const TemplateList &data)
    {
        getWords->train(data);
        TemplateList bins;
        getWords->project(data, bins);

        numWords = 0;
        foreach (const Template &t, bins) {
            double minVal, maxVal;
            minMaxLoc(t, &minVal, &maxVal);
            numWords = max(numWords, int(maxVal)+1);
        }

        TemplateList reworded; reworded.reserve(data.size());
        foreach (const Template &t, data)
            reworded.append(reword(t));
        byWord->train(reworded);
    }

    void project(const Template &src, Template &dst) const
    {
        byWord->project(reword(src), dst);
    }

    Template reword(const Template &src) const
    {
        Template words;
        getWords->project(src, words);
        QVector<int> wordCounts(numWords, 0);
        for (int i=0; i<words.m().rows; i++)
            wordCounts[words.m().at<uchar>(i,0)]++;
        Template reworded(src.file); reworded.reserve(numWords);
        for (int i=0; i<numWords; i++)
            reworded.append(Mat(wordCounts[i], src.m().cols, src.m().type()));
        QVector<int> indicies(numWords, 0);
        for (int i=0; i<src.m().rows; i++) {
            const int word = words.m().at<uchar>(i,0);
            src.m().row(i).copyTo(reworded[word].row(indicies[word]++));
        }
        return reworded;
    }
};

BR_REGISTER(Transform, WordWiseTransform)

} // namespace br

#include "integral.moc"
