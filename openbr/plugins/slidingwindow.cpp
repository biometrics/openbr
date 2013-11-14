#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"
#include "openbr/core/qtutils.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace Eigen;

namespace br
{

// Find avg aspect ratio
static float getAspectRatio(const TemplateList &data)
{
    double tempRatio = 0;
    int ratioCnt = 0;

    foreach (const Template &tmpl, data) {
        QList<Rect> posRects = OpenCVUtils::toRects(tmpl.file.rects());
        foreach (const Rect &posRect, posRects) {
            if (posRect.x + posRect.width >= tmpl.m().cols || posRect.y + posRect.height >= tmpl.m().rows || posRect.x < 0 || posRect.y < 0) {
                continue;
            }
            tempRatio += (float)posRect.width / (float)posRect.height;
            ratioCnt += 1;
        }
    }
    return tempRatio / (double)ratioCnt;
}

/*!
 * \ingroup transforms
 * \brief Applies a transform to a sliding window.
 *        Discards negative detections.
 * \author Austin Blanton \cite imaus10
 */
class SlidingWindowTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int stepSize READ get_stepSize WRITE set_stepSize RESET reset_stepSize STORED false)
    Q_PROPERTY(bool takeFirst READ get_takeFirst WRITE set_takeFirst RESET reset_takeFirst STORED false)
    Q_PROPERTY(int windowWidth READ get_windowWidth WRITE set_windowWidth RESET reset_windowWidth STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(int, stepSize, 1)
    BR_PROPERTY(bool, takeFirst, false)
    BR_PROPERTY(int, windowWidth, 24)
    BR_PROPERTY(float, threshold, 0)

private:
    int windowHeight;

    void train(const TemplateList &data)
    {
        float aspectRatio = data.first().file.get<float>("aspectRatio", -1);
        if (aspectRatio == -1)
            aspectRatio = getAspectRatio(data);
        windowHeight = qRound(windowWidth / aspectRatio);
        if (transform->trainable) {
            transform->train(data);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        // no need to slide a window over ground truth data
        if (src.file.getBool("Train", false)) return;

        dst.file.clearRects();
        float scale = src.file.get<float>("scale", 1);
        Template windowTemplate(src.file, src);
        QList<float> confidences = dst.file.getList<float>("Confidences", QList<float>());
        for (double y = 0; y + windowHeight < src.m().rows; y += stepSize) {
            for (double x = 0; x + windowWidth < src.m().cols; x += stepSize) {
                Mat windowMat(src, Rect(x, y, windowWidth, windowHeight));
                windowTemplate.replace(0,windowMat);
                Template detect;
                transform->project(windowTemplate, detect);
                float conf = detect.m().at<float>(0);

                // the result will be in the Label
                if (conf > threshold) {
                    dst.file.appendRect(QRectF((float) x * scale, (float) y * scale, (float) windowWidth * scale, (float) windowHeight * scale));
                    confidences.append(conf);
                    if (takeFirst)
                        return;
                }
            }
        }
        dst.file.setList<float>("Confidences", confidences);
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
        stream << windowHeight;
    }

    void load(QDataStream &stream)
    {
        transform->load(stream);
        stream >> windowHeight;
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

static TemplateList cropTrainingSamples(const TemplateList &data, const float aspectRatio, const int minSize = 0, const float maxOverlap = 0.5, const int negToPosRatio = 1)
{
    TemplateList result;
    foreach (const Template &tmpl, data) {
        QList<Rect> posRects = OpenCVUtils::toRects(tmpl.file.rects());
        QList<Rect> negRects;
        for (int i=0; i<posRects.size(); i++) {
            Rect &posRect = posRects[i];

            // Adjust for training samples that have different aspect ratios
            const int diff = posRect.width - int(posRect.height * aspectRatio);
            posRect.x += diff / 2;
            posRect.width += diff;

            // Ignore samples larger than the image
            if ((posRect.x + posRect.width >= tmpl.m().cols) ||
                (posRect.y + posRect.height >= tmpl.m().rows) ||
                (posRect.x < 0) ||
                (posRect.y < 0))
                continue;

            result += Template(tmpl.file, Mat(tmpl, posRect));

            // Add random negative samples
            Mat m = tmpl.m();
            int sample = 0;
            while (sample < negToPosRatio) {
                const int x = rand() % m.cols;
                const int y = rand() % m.rows;
                const int maxWidth = m.cols - x;
                const int maxHeight = m.rows - y;
                if (maxWidth <= minSize || maxHeight <= minSize)
                    continue;

                int height;
                int width;
                if (aspectRatio > (float) maxWidth / (float) maxHeight) {
                    width = rand() % (maxWidth - minSize) + minSize;
                    height = qRound(width / aspectRatio);
                } else {
                    height = rand() % (maxHeight - minSize) + minSize;
                    width = qRound(height * aspectRatio);
                }
                Rect negRect(x, y, width, height);

                // The negative samples cannot overlap the positive samples at
                // all, but they may partially overlap with other negatives.
                if (OpenCVUtils::overlaps(posRects, negRect, 0) ||
                    OpenCVUtils::overlaps(negRects, negRect, maxOverlap))
                    continue;

                result += Template(tmpl.file, Mat(tmpl, negRect));
                result.last().file.set("Label", QString("neg"));
                sample++;
            }
        }
    }

    return result;
}

/*!
 * \ingroup transforms
 * \brief .
 * \author Austin Blanton \cite imaus10
 */
class BuildScalesTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(double scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(bool takeLargestScale READ get_takeLargestScale WRITE set_takeLargestScale RESET reset_takeLargestScale STORED false)
    Q_PROPERTY(int windowWidth READ get_windowWidth WRITE set_windowWidth RESET reset_windowWidth STORED false)
    Q_PROPERTY(int negToPosRatio READ get_negToPosRatio WRITE set_negToPosRatio RESET reset_negToPosRatio STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(double maxOverlap READ get_maxOverlap WRITE set_maxOverlap RESET reset_maxOverlap STORED false)
    Q_PROPERTY(float minScale READ get_minScale WRITE set_minScale RESET reset_minScale STORED false)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(double, scaleFactor, 0.75)
    BR_PROPERTY(bool, takeLargestScale, false)
    BR_PROPERTY(int, windowWidth, 24)
    BR_PROPERTY(int, negToPosRatio, 1)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(double, maxOverlap, 0)
    BR_PROPERTY(float, minScale, 1.0)

private:
    float aspectRatio;
    int windowHeight;

    void train(const TemplateList &data)
    {
        aspectRatio = getAspectRatio(data);
        windowHeight = qRound(windowWidth / aspectRatio);
        if (transform->trainable) {
            TemplateList full;
            foreach (const Template &roi, cropTrainingSamples(data, aspectRatio, minSize, maxOverlap, negToPosRatio)) {
                Mat resized;
                resize(roi, resized, Size(windowWidth, windowHeight));
                full += Template(roi.file, resized);
            }
            full.first().file.set("aspectRatio", aspectRatio);
            transform->train(full);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        // do not scale images during training
        if (src.file.getBool("Train", false)) return;

        int rows = src.m().rows;
        int cols = src.m().cols;
        int windowHeight = (int) qRound((float) windowWidth / aspectRatio);
        float startScale;
        if ((cols / rows) > aspectRatio)
            startScale = qRound((float) rows / (float) windowHeight);
        else
            startScale = qRound((float) cols / (float) windowWidth);
        for (float scale = startScale; scale >= minScale; scale -= (1.0 - scaleFactor)) {
            Template scaleImg(src.file, Mat());
            scaleImg.file.set("scale", scale);
            resize(src, scaleImg, Size(qRound(cols / scale), qRound(rows / scale)));
            transform->project(scaleImg, dst);
            if (takeLargestScale && !dst.file.rects().empty())
                return;
        }
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
        stream << aspectRatio << windowHeight;
    }
    void load(QDataStream &stream)
    {
        transform->load(stream);
        stream >> aspectRatio >> windowHeight;
    }
};

BR_REGISTER(Transform, BuildScalesTransform)

/*!
 * \ingroup transforms
 * \brief Sample detection bounding boxes from without resizing
 * \author Josh Klontz \cite jklontz
 */
class Detector : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, make("Identity"))

    void train(const TemplateList &data)
    {
        const float aspectRatio = getAspectRatio(data);
        TemplateList cropped = cropTrainingSamples(data, aspectRatio);
        qDebug("Detector using: %d training samples.", cropped.size());
        cropped.first().file.set("aspectRatio", aspectRatio);
        transform->train(cropped);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};

BR_REGISTER(Transform, Detector)

/*!
 * \ingroup transforms
 * \brief Detects objects with OpenCV's built-in HOG detection.
 * \author Austin Blanton \cite imaus10
 */
class HOGDetectTransform : public UntrainableTransform
{
    Q_OBJECT

    HOGDescriptor hog;

    void init()
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        std::vector<Rect> objLocs;
        QList<Rect> rects;
        hog.detectMultiScale(src, objLocs);
        foreach (const Rect &obj, objLocs)
            rects.append(obj);
        dst.file.setRects(rects);
    }
};

BR_REGISTER(Transform, HOGDetectTransform)

/*!
 * \ingroup transforms
 * \brief Consolidate redundant/overlapping detections.
 * \author Brendan Klare \cite bklare
 */
class ConsolidateDetectionsTransform : public Transform
{
    Q_OBJECT

public:
    ConsolidateDetectionsTransform() : Transform(false, false) {}
private:

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (!dst.file.contains("Confidences"))
            return;

        //Compute overlap between rectangles and create discrete Laplacian matrix
        QList<Rect> rects = OpenCVUtils::toRects(src.file.rects());
        int n = rects.size();
        if (n == 0)
            return;
        MatrixXf laplace(n,n);
        for (int i = 0; i < n; i++) {
            laplace(i,i) = 0;
        }
        for (int i = 0; i < n; i++){
            for (int j = i + 1; j < n; j++) {
                float overlap = (float)((rects[i] & rects[j]).area()) / (float)max(rects[i].area(), rects[j].area());
                if (overlap > 0.5) {
                    laplace(i,j) = -1.0;
                    laplace(j,i) = -1.0;
                    laplace(i,i) = laplace(i,i) + 1.0;
                    laplace(j,j) = laplace(j,j) + 1.0;
                } else {
                    laplace(i,j) = 0;
                    laplace(j,i) = 0;
                }
            }
        }

        // Compute eigendecomposition
        SelfAdjointEigenSolver<Eigen::MatrixXf> eSolver(laplace);
        MatrixXf allEVals = eSolver.eigenvalues();
        MatrixXf allEVecs = eSolver.eigenvectors();

        //Keep eigenvectors with zero eigenvalues
        int nRegions = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(allEVals(i)) < 1e-4) {
                nRegions++;
            }
        }
        MatrixXf regionVecs(n, nRegions);
        for (int i = 0, cnt = 0; i < n; i++) {
            if (fabs(allEVals(i)) < 1e-4)
                regionVecs.col(cnt++) = allEVecs.col(i);
        }

        //Determine membership for each consolidated location
        // and compute average of regions. This is determined by
        // finding which eigenvector has the highest magnitude for
        // each input dimension. Each input dimension corresponds to
        // one of the input rect region. Thus, each eigenvector represents
        // a set of overlaping regions.
        float midX[nRegions];
        float midY[nRegions];
        float avgWidth[nRegions];
        float avgHeight[nRegions];
        float confs[nRegions];
        int cnts[nRegions];
        int mx;
        int mxIdx;
        for (int i = 0 ; i < nRegions; i++) {
            midX[i] = 0;
            midY[i] = 0;
            avgWidth[i] = 0;
            avgHeight[i] = 0;
            confs[i] = 0;
            cnts[i] = 0;
        }

        QList<float> confidences = dst.file.getList<float>("Confidences");
        for (int i = 0; i < n; i++) {
            mx = 0.0;
            mxIdx = -1;

            for (int j = 0; j < nRegions; j++) {
                if (fabs(regionVecs(i,j)) > mx) {
                    mx = fabs(regionVecs(i,j));
                    mxIdx = j;
                }
            }

            Rect curRect = rects[i];
            midX[mxIdx] += ((float)curRect.x + (float)curRect.width  / 2.0);
            midY[mxIdx] += ((float)curRect.y + (float)curRect.height / 2.0);
            avgWidth[mxIdx]  += (float) curRect.width;
            avgHeight[mxIdx] += (float) curRect.height;
            confs[mxIdx] += confidences[i];
            cnts[mxIdx]++;
        }

        QList<Rect> consolidatedRects;
        QList<float> consolidatedConfidences;
        for (int i = 0; i < nRegions; i++) {
            float cntF = (float) cnts[i];
            if (cntF > 0) {
                int x = qRound((midX[i] / cntF) - (avgWidth[i] / cntF) / 2.0);
                int y = qRound((midY[i] / cntF) - (avgHeight[i] / cntF) / 2.0);
                int w = qRound(avgWidth[i] / cntF);
                int h = qRound(avgHeight[i] / cntF);
                consolidatedRects.append(Rect(x,y,w,h));
                consolidatedConfidences.append(confs[i] / cntF);
            }
        }

        dst.file.setRects(consolidatedRects);
        dst.file.setList<float>("Confidences", consolidatedConfidences);
    }
};

BR_REGISTER(Transform, ConsolidateDetectionsTransform)

} // namespace br

#include "slidingwindow.moc"
