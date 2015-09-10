#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

#define CV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x + w, y) */                                                      \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
    /* (x + w, y) */                                                      \
    (p2) = (rect).x + (step) * ((rect).y + (rect).height);                \
    /* (x + w, y + h) */                                                  \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

/*!
 * \brief An implementation of Haar Features for Viola-Jones cascade object detection
 * \author Jordan Cheney \cite jcheney
 * \br_property int winWidth The width of the input image. The total feature space is based on this and the winHeight
 * \br_property int winHeight The height of the input image. The total feature space is based on this and the winWidth.
 * \br_paper Paul Viola, Michael Jones
 *           Rapid Object Detection using a Boosted Cascade of Simple Features
 *           CVPR, 2001
 * \br_link Rapid Object Detection using a Boosted Cascade of Simple Features https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
 */
class HaarRepresentation : public Representation
{
    Q_OBJECT

    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)

    void init()
    {
        if (features.isEmpty()) {
            int offset = winWidth + 1;
            for (int x = 0; x < winWidth; x++) {
                for (int y = 0; y < winHeight; y++) {
                    for (int dx = 1; dx <= winWidth; dx++) {
                        for (int dy = 1; dy <= winHeight; dy++) {
                            // haar_x2
                            if ((x+dx*2 <= winWidth) && (y+dy <= winHeight))
                                features.append(Feature(offset,
                                                        x,    y, dx*2, dy, -1,
                                                        x+dx, y, dx  , dy, +2));
                            // haar_y2
                            if ((x+dx <= winWidth) && (y+dy*2 <= winHeight))
                                features.append(Feature(offset,
                                                        x,    y, dx, dy*2, -1,
                                                        x, y+dy, dx, dy,   +2));
                            // haar_x3
                            if ((x+dx*3 <= winWidth) && (y+dy <= winHeight))
                                features.append(Feature(offset,
                                                        x,    y, dx*3, dy, -1,
                                                        x+dx, y, dx  , dy, +3));
                            // haar_y3
                            if ((x+dx <= winWidth) && (y+dy*3 <= winHeight))
                                features.append(Feature(offset,
                                                        x, y,    dx, dy*3, -1,
                                                        x, y+dy, dx, dy,   +3));
                            // x2_y2
                            if ((x+dx*2 <= winWidth) && (y+dy*2 <= winHeight))
                                features.append(Feature(offset,
                                                        x,    y,    dx*2, dy*2, -1,
                                                        x,    y,    dx,   dy,   +2,
                                                        x+dx, y+dy, dx,   dy,   +2));


                        }
                    }
                }
            }
        }
    }

    void preprocess(const Mat &src, Mat &dst) const
    {
        integral(src, dst);
    }

    float evaluate(const Mat &image, int idx) const
    {
        return (float)features[idx].calc(image);
    }

    Mat evaluate(const Mat &image, const QList<int> &indices) const
    {
        int size = indices.empty() ? numFeatures() : indices.size();

        Mat result(1, size, CV_32FC1);
        for (int i = 0; i < size; i++)
            result.at<float>(i) = evaluate(image, indices.empty() ? i : indices[i]);
        return result;
    }

    int numFeatures() const
    {
        return features.size();
    }

    Size windowSize(int *dx, int *dy) const
    {
        if (dx && dy)
            *dx = *dy = 1;
        return Size(winWidth, winHeight);
    }

    int maxCatCount() const { return 0; }

    struct Feature
    {
        Feature();
        Feature( int offset,
            int x0, int y0, int w0, int h0, float wt0,
            int x1, int y1, int w1, int h1, float wt1,
            int x2 = 0, int y2 = 0, int w2 = 0, int h2 = 0, float wt2 = 0.0F );
        float calc(const Mat &img) const;

        struct {
            Rect r;
            float weight;
        } rect[3];

        struct {
            int p0, p1, p2, p3;
        } fastRect[3];
    };

    QList<Feature> features;
};

BR_REGISTER(Representation, HaarRepresentation)

HaarRepresentation::Feature::Feature()
{
    rect[0].r = rect[1].r = rect[2].r = Rect(0,0,0,0);
    rect[0].weight = rect[1].weight = rect[2].weight = 0;
}

HaarRepresentation::Feature::Feature(int offset,
            int x0, int y0, int w0, int h0, float wt0,
            int x1, int y1, int w1, int h1, float wt1,
            int x2, int y2, int w2, int h2, float wt2)
{
    rect[0].r.x = x0;
    rect[0].r.y = y0;
    rect[0].r.width  = w0;
    rect[0].r.height = h0;
    rect[0].weight   = wt0;

    rect[1].r.x = x1;
    rect[1].r.y = y1;
    rect[1].r.width  = w1;
    rect[1].r.height = h1;
    rect[1].weight   = wt1;

    rect[2].r.x = x2;
    rect[2].r.y = y2;
    rect[2].r.width  = w2;
    rect[2].r.height = h2;
    rect[2].weight   = wt2;

    for (int j = 0; j < 3; j++) {
        if( rect[j].weight == 0.0F )
            break;
        CV_SUM_OFFSETS(fastRect[j].p0, fastRect[j].p1, fastRect[j].p2, fastRect[j].p3, rect[j].r, offset)
    }
}

inline float HaarRepresentation::Feature::calc(const Mat &img) const
{
    const int* ptr = img.ptr<int>();
    float ret = rect[0].weight * (ptr[fastRect[0].p0] - ptr[fastRect[0].p1] - ptr[fastRect[0].p2] + ptr[fastRect[0].p3]) +
                rect[1].weight * (ptr[fastRect[1].p0] - ptr[fastRect[1].p1] - ptr[fastRect[1].p2] + ptr[fastRect[1].p3]);
    if (rect[2].weight != 0.0f)
        ret += rect[2].weight * (ptr[fastRect[2].p0] - ptr[fastRect[2].p1] - ptr[fastRect[2].p2] + ptr[fastRect[2].p3]);
    return ret;
}

} // namespace br

#include "representation/haar.moc"

