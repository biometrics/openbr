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
 * \brief Concatenates all input matrices into a single matrix.
 * \author Josh Klontz \cite jklontz
 */
class CatTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int partitions READ get_partitions WRITE set_partitions RESET reset_partitions)
    BR_PROPERTY(int, partitions, 1)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;

        if (src.size() % partitions != 0)
            qFatal("%d partitions does not evenly divide %d matrices.", partitions, src.size());
        QVector<int> sizes(partitions, 0);
        for (int i=0; i<src.size(); i++)
            sizes[i%partitions] += src[i].total();

        if (!src.empty())
            foreach (int size, sizes)
                dst.append(Mat(1, size, src.m().type()));

        QVector<int> offsets(partitions, 0);
        for (int i=0; i<src.size(); i++) {
            size_t size = src[i].total() * src[i].elemSize();
            int j = i % partitions;
            memcpy(&dst[j].data[offsets[j]], src[i].ptr(), size);
            offsets[j] += size;
        }
    }
};

BR_REGISTER(Transform, CatTransform)

} // namespace br

#include "imgproc/cat.moc"
