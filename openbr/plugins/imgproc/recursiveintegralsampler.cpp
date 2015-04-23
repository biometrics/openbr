/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Construct Template in a recursive decent manner.
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

    static void integralHistogram(const Mat &src, const int x, const int y, const int width, const int height, Mat &dst, int index)
    {
        const int channels = src.channels();
        OutputDescriptor(dst.ptr<float>(index), channels, 1) =
            (  InputDescriptor(src.ptr<qint32>(y+height, x+width), channels, 1)
             - InputDescriptor(src.ptr<qint32>(y,        x+width), channels, 1)
             - InputDescriptor(src.ptr<qint32>(y+height, x),       channels, 1)
             + InputDescriptor(src.ptr<qint32>(y,        x),       channels, 1)).cast<float>()/(height*width);
    }

    void computeDescriptor(const Mat &src, Mat &dst) const
    {
        const int channels = src.channels();
        const int rows = src.rows-1; // Integral images have an extra row and column
        const int columns = src.cols-1;

        Mat tmp(5, channels, CV_32FC1);
        integralHistogram(src,         0,      0, columns/2, rows/2, tmp, 0);
        integralHistogram(src, columns/2,      0, columns/2, rows/2, tmp, 1);
        integralHistogram(src,         0, rows/2, columns/2, rows/2, tmp, 2);
        integralHistogram(src, columns/2, rows/2, columns/2, rows/2, tmp, 3);
        integralHistogram(src, columns/4, rows/4, columns/2, rows/2, tmp, 4);
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
        dst = dst.reshape(1, 1);
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
        if (src.first().m().depth() != CV_32S)
            qFatal("Expected CV_32S depth!");

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

} // namespace br

#include "imgproc/recursiveintegralsampler.moc"
