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

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Resize the template
 * \author Josh Klontz \cite jklontz
 * \br_property enum method Resize method. Good options are: [Area should be used for shrinking an image, Cubic for slow but accurate enlargment, Bilin for fast enlargement]
 * \br_property bool preserveAspect If true, the image will be sized per specification, but a border will be applied to preserve aspect ratio.
 */
class ResizeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Method)

public:
    /*!< */
    enum Method { Near = INTER_NEAREST,
                  Area = INTER_AREA,
                  Bilin = INTER_LINEAR,
                  Cubic = INTER_CUBIC,
                  Lanczo = INTER_LANCZOS4};

private:
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)
    Q_PROPERTY(bool preserveAspect READ get_preserveAspect WRITE set_preserveAspect RESET reset_preserveAspect STORED false)
    BR_PROPERTY(int, rows, -1)
    BR_PROPERTY(int, columns, -1)
    BR_PROPERTY(Method, method, Bilin)
    BR_PROPERTY(bool, preserveAspect, false)

    void project(const Template &src, Template &dst) const
    {
        if (!preserveAspect)
            resize(src, dst, Size((columns == -1) ? src.m().cols*rows/src.m().rows : columns, rows), 0, 0, method);
        else {
            float inRatio = (float) src.m().rows / src.m().cols;
            float outRatio = (float) rows / columns;
            dst = Mat::zeros(rows, columns, src.m().type());
            if (outRatio > inRatio) {
                float heightAR = src.m().rows * inRatio / outRatio;
                Mat buffer;
                resize(src, buffer, Size(columns, heightAR), 0, 0, method);
                buffer.copyTo(dst.m()(Rect(0, (rows - heightAR) / 2, columns, heightAR)));
            } else {
                float widthAR = src.m().cols / inRatio * outRatio;
                Mat buffer;
                resize(src, buffer, Size(widthAR, rows), 0, 0, method);
                buffer.copyTo(dst.m()(Rect((columns - widthAR) / 2, 0, widthAR, rows)));
            }
        }
    }
};

BR_REGISTER(Transform, ResizeTransform)

} // namespace br

#include "imgproc/resize.moc"
