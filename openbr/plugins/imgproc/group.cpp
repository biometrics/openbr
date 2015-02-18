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

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Group all input matrices into a single matrix.
 *
 * Similar to CatTransfrom but groups every _size_ adjacent matricies.
 * \author Josh Klontz \cite jklontz
 */
class GroupTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int size READ get_size WRITE set_size RESET reset_size STORED false)
    BR_PROPERTY(int, size, 1)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;

        if (src.size() % size != 0)
            qFatal("%d size does not evenly divide %d matrices.", size, src.size());
        QVector<int> sizes(src.size() / size, 0);
        for (int i=0; i<src.size(); i++)
            sizes[i/size] += src[i].total();

        if (!src.empty())
            foreach (int size_, sizes)
                dst.append(Mat(1, size_, src.m().type()));

        QVector<int> offsets(src.size() / size, 0);
        for (int i=0; i<src.size(); i++) {
            const size_t bytes = src[i].total() * src[i].elemSize();
            const int j = i / size;
            memcpy(&dst[j].data[offsets[j]], src[i].ptr(), bytes);
            offsets[j] += bytes;
        }
    }
};

BR_REGISTER(Transform, GroupTransform)

} // namespace br

#include "imgproc/group.moc"
