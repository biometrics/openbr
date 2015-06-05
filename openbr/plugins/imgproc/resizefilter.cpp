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
 * \brief Resize the template depending on its metadata
 * \author Jordan Cheney \cite JordanCheney
 * \note Method: Area should be used for shrinking an image, Cubic for slow but accurate enlargment, Bilin for fast enlargement.
 */
class ResizeFilterTransform : public UntrainableTransform
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
    Q_PROPERTY(QString filterKey READ get_filterKey WRITE set_filterKey RESET reset_filterKey STORED false)
    Q_PROPERTY(QString filterVal READ get_filterVal WRITE set_filterVal RESET reset_filterVal STORED false)
    BR_PROPERTY(int, rows, -1)
    BR_PROPERTY(int, columns, -1)
    BR_PROPERTY(Method, method, Bilin)
    BR_PROPERTY(QString, filterKey, "Label")
    BR_PROPERTY(QString, filterVal, "1.0")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (src.file.get<QString>(filterKey) == filterVal)
            resize(src, dst, Size((columns == -1) ? src.m().cols*rows/src.m().rows : columns, rows), 0, 0, method);
    }
};

BR_REGISTER(Transform, ResizeFilterTransform)

} // namespace br

#include "imgproc/resizefilter.moc"
