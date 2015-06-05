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
 * \brief An implementation of MBLBP as an OpenBR Representation
 * \author Jordan Cheney \cite jcheney
 * \br_property int winWidth The width of the input image. The total feature space is based on this and the winHeight
 * \br_property int winHeight The height of the input image. The total feature space is based on this and the winWidth.
 * \br_paper Shengcai Liao, Xiangxin Zhu, Zhen Lei, Lun Zhang, Stan Z. Li
 *           Learning Multi-scale Block Local Binary Patterns for Face Recognition
 *           ICB, 2007
 * \br_link Learning Multi-scale Block Local Binary Patterns for Face Recognition http://www.cbsr.ia.ac.cn/users/lzhang/papers/ICB07/ICB07_Liao.pdf
 */
class MBLBPRepresentation : public Representation
{
    Q_OBJECT

    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)

    void init()
    {
        int offset = winWidth + 1;
        for (int x = 0; x < winWidth; x++ )
            for (int y = 0; y < winHeight; y++ )
                for (int w = 1; w <= winWidth / 3; w++ )
                    for (int h = 1; h <= winHeight / 3; h++ )
                        if ((x+3*w <= winWidth) && (y+3*h <= winHeight) )
                            features.append(Feature(offset, x, y, w, h ) );
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

    Size windowSize(int *dx, int *dy) const
    {
        if (dx && dy)
            *dx = *dy = 1;
        return Size(winWidth, winHeight);
    }

    int numFeatures() const { return features.size(); }
    int maxCatCount() const { return 256; }

    struct Feature
    {
        Feature() { rect = Rect(0, 0, 0, 0); }
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc(const Mat &img) const;

        Rect rect;
        int p[16];
    };
    QList<Feature> features;
};

BR_REGISTER(Representation, MBLBPRepresentation)

static inline void calcOffset(int &p0, int &p1, int &p2, int &p3, Rect rect, int offset)
{
    /* (x, y) */
    p0 = rect.x + offset * rect.y;
    /* (x + w, y) */
    p1 = rect.x + rect.width + offset * rect.y;
    /* (x + w, y) */
    p2 = rect.x + offset * (rect.y + rect.height);
    /* (x + w, y + h) */
    p3 = rect.x + rect.width + offset * (rect.y + rect.height);
}

MBLBPRepresentation::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight )
{
    Rect tr = rect = cvRect(x, y, _blockWidth, _blockHeight);
    calcOffset(p[0], p[1], p[4], p[5], tr, offset);
    tr.x += 2*rect.width;
    calcOffset(p[2], p[3], p[6], p[7], tr, offset);
    tr.y +=2*rect.height;
    calcOffset(p[10], p[11], p[14], p[15], tr, offset);
    tr.x -= 2*rect.width;
    calcOffset(p[8], p[9], p[12], p[13], tr, offset);
}

inline uchar MBLBPRepresentation::Feature::calc(const Mat &img) const
{
    const int* ptr = img.ptr<int>();
    int cval = ptr[p[5]] - ptr[p[6]] - ptr[p[9]] + ptr[p[10]];

    return (uchar)((ptr[p[0]] - ptr[p[1]] - ptr[p[4]] + ptr[p[5]] >= cval ? 128 : 0) |   // 0
                   (ptr[p[1]] - ptr[p[2]] - ptr[p[5]] + ptr[p[6]] >= cval ? 64 : 0) |    // 1
                   (ptr[p[2]] - ptr[p[3]] - ptr[p[6]] + ptr[p[7]] >= cval ? 32 : 0) |    // 2
                   (ptr[p[6]] - ptr[p[7]] - ptr[p[10]] + ptr[p[11]] >= cval ? 16 : 0) |  // 5
                   (ptr[p[10]] - ptr[p[11]] - ptr[p[14]] + ptr[p[15]] >= cval ? 8 : 0) | // 8
                   (ptr[p[9]] - ptr[p[10]] - ptr[p[13]] + ptr[p[14]] >= cval ? 4 : 0) |  // 7
                   (ptr[p[8]] - ptr[p[9]] - ptr[p[12]] + ptr[p[13]] >= cval ? 2 : 0) |   // 6
                   (ptr[p[4]] - ptr[p[5]] - ptr[p[8]] + ptr[p[9]] >= cval ? 1 : 0));     // 3
}

} // namespace br

#include "representation/mblbp.moc"
