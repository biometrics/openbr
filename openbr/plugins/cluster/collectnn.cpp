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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Collect nearest neighbors and append them to metadata.
 * \author Charles Otto \cite caotto
 * \br_property int keep The maximum number of nearest neighbors to keep. Default is 20.
 */
class CollectNNTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(int keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    BR_PROPERTY(int, keep, 20)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.clear();
        dst.m() = cv::Mat();
        Neighbors neighbors;
        for (int i=0; i < src.m().cols;i++) {
            // skip self compares
            if (i == src.file.get<int>("FrameNumber"))
                continue;
            neighbors.append(Neighbor(i, src.m().at<float>(0,i)));
        }
        int actuallyKeep = std::min(keep, neighbors.size());
        std::partial_sort(neighbors.begin(), neighbors.begin()+actuallyKeep, neighbors.end(), compareNeighbors);

        Neighbors selected = neighbors.mid(0, actuallyKeep);
        dst.file.set("neighbors", QVariant::fromValue(selected));
    }
};

BR_REGISTER(Transform, CollectNNTransform)

} // namespace br

#include "cluster/collectnn.moc"
