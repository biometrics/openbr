#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup galleries
 * \brief Computes first order gradient histogram features using an integral image
 * \author Scott Klum \cite sklum
 */
class GradientHistogramRepresentation : public Representation
{
    Q_OBJECT

    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int bins READ get_bins WRITE set_bins RESET reset_bins STORED false)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, bins, 6)

    void init()
    {
        int dx, dy;
        Size size = windowSize(&dx,&dy);

        int width = size.width+dx, height = size.height+dy;

        // Enumerate all possible rectangles
        for (int x=0; x<width; x++)
            for (int y=0; y<height; y++)
                for (int w=1; w <= width-x; w++)
                    for (int h=1; h <= height-y; h++)
                        features.append(Rect(x,y,width,height));
    }

    void preprocess(const Mat &src, Mat &dst) const
    {

        // Compute as is done in GradientTransform
        Mat dx, dy, magnitude, angle;
        Sobel(src, dx, CV_32F, 1, 0, CV_SCHARR);
        Sobel(src, dy, CV_32F, 0, 1, CV_SCHARR);
        cartToPolar(dx, dy, magnitude, angle, true);

        const double floor = ((src.depth() == CV_32F) || (src.depth() == CV_64F)) ? -0.5 : 0;

        Mat histogram;
        angle.convertTo(histogram, bins > 256 ? CV_16U : CV_8U, bins/360., floor);

        // Mask and compute integral image
        std::vector<Mat> outputs;
        for (int i=0; i<1; i++) {
            Mat output = (histogram == i);
            Mat integralImg;
            integral(output, integralImg);
            outputs.push_back(integralImg);
        }

        merge(outputs,dst);
    }

    /*  ___ ___
     * |   |   |
     * | A | B |
     * |___|___|
     * |   |   |
     * | C | D |
     * |___|___|
     *
     * 1, 2, 3 and 4 refer to the lower right corners of A, B, C, and D, respectively.
     * Rectangle D can be computed as 4 + 1 - (2 + 3)
     */

    float evaluate(const Mat &image, int idx) const
    {
        /* Stored in memory as (row,col,channel): (0,0,0), (0,0,1), ... , (0,0,bin-1), (0,1,0), (0,1,1), ... , (0,1,bin-1), ... , (0,cols,0), (0,cols,1), ... , (0,cols,bin-1)
         *                                        (1,0,0), (1,0,1), ... , (1,0,bin-1), (1,1,0), (1,1,1), ... , (1,1,bin-1), ... , (1,cols,0), (1,cols,1), ... , (1,cols,bin-1)
         *
         *                                        (row,0,0), (row,0,1), ... , (row,0,bin-1), (row,1,0), (row,1,1), ... , (row,1,bin-1), ... , (row,cols,0), (row,cols,1), ... , (row,cols,bin-1)
         */

        // To which channel does an index belong?
        const int index = idx % features.size();
        const int channel = idx / features.size();

        int four = image.ptr<int>(features[index].y+features[index].height)[(features[index].x+features[index].width)*channel];
        int one = image.ptr<int>(features[index].y)[features[index].x*channel];
        int two = image.ptr<int>(features[index].y)[(features[index].x+features[index].width)*channel];
        int three = image.ptr<int>(features[index].y+features[index].height)[features[index].x*channel];

        return four + one - (two + three);
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

    QList<Rect> features;
};

BR_REGISTER(Representation, GradientHistogramRepresentation)

} // namespace br

#include "representation/gradienthistogram.moc"


