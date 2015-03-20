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
 * \brief Enforce the matrix has a certain number of channels by adding or removing channels.
 * \author Josh Klontz \cite jklontz
 */
class EnsureChannelsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() == n) {
            dst = src;
        } else {
            std::vector<Mat> mv;
            split(src, mv);

            // Add extra channels
            while ((int)mv.size() < n) {
                for (int i=0; i<src.m().channels(); i++) {
                    mv.push_back(mv[i]);
                    if ((int)mv.size() == n)
                        break;
                }
            }

            // Remove extra channels
            while ((int)mv.size() > n)
                mv.pop_back();

            merge(mv, dst);
        }
    }
};

BR_REGISTER(Transform, EnsureChannelsTransform)

} // namespace br

#include "imgproc/ensurechannels.moc"
