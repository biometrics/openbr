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
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <limits>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert the image into a feature vector using Local Binary Patterns
 * \br_paper Ahonen, T.; Hadid, A.; Pietikainen, M.;
 *           "Face Description with Local Binary Patterns: Application to Face Recognition"
 *           Pattern Analysis and Machine Intelligence, IEEE Transactions, vol.28, no.12, pp.2037-2041, Dec. 2006
 * \author Josh Klontz \cite jklontz
 */
class LBPTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    Q_PROPERTY(int maxTransitions READ get_maxTransitions WRITE set_maxTransitions RESET reset_maxTransitions STORED false)
    Q_PROPERTY(bool rotationInvariant READ get_rotationInvariant WRITE set_rotationInvariant RESET reset_rotationInvariant STORED false)
    BR_PROPERTY(int, radius, 1)
    BR_PROPERTY(int, maxTransitions, 8)
    BR_PROPERTY(bool, rotationInvariant, false)

    uchar lut[256];
    uchar null;

    /* Returns the number of 0->1 or 1->0 transitions in i */
    static int numTransitions(int i)
    {
        int transitions = 0;
        int curParity = i%2;
        for (int j=1; j<=8; j++) {
            int parity = (i>>(j%8)) % 2;
            if (parity != curParity) transitions++;
            curParity = parity;
        }
        return transitions;
    }

    static int rotationInvariantEquivalent(int i)
    {
        int min = std::numeric_limits<int>::max();
        for (int j=0; j<8; j++) {
            bool parity = i % 2;
            i = i >> 1;
            if (parity) i+=128;
            min = std::min(min, i);
        }
        return min;
    }

    void init()
    {
        bool set[256];
        uchar uid = 0;
        for (int i=0; i<256; i++) {
            if (numTransitions(i) <= maxTransitions) {
                int id;
                if (rotationInvariant) {
                    int rie = rotationInvariantEquivalent(i);
                    if (i == rie) id = uid++;
                    else          id = lut[rie];
                } else            id = uid++;
                lut[i] = id;
                set[i] = true;
            } else {
                set[i] = false;
            }
        }

        null = uid;
        for (int i=0; i<256; i++)
            if (!set[i])
                lut[i] = null; // Set to null id
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m; src.m().convertTo(m, CV_32F); assert(m.isContinuous() && (m.channels() == 1));
        Mat n(m.rows, m.cols, CV_8UC1);
        n = null; // Initialize to NULL LBP pattern

        const float *p = (const float*)m.ptr();
        for (int r=radius; r<m.rows-radius; r++) {
            for (int c=radius; c<m.cols-radius; c++) {
                const float cval  =     (p[(r+0*radius)*m.cols+c+0*radius]);
                n.at<uchar>(r, c) = lut[(p[(r-1*radius)*m.cols+c-1*radius] >= cval ? 128 : 0) |
                                        (p[(r-1*radius)*m.cols+c+0*radius] >= cval ? 64  : 0) |
                                        (p[(r-1*radius)*m.cols+c+1*radius] >= cval ? 32  : 0) |
                                        (p[(r+0*radius)*m.cols+c+1*radius] >= cval ? 16  : 0) |
                                        (p[(r+1*radius)*m.cols+c+1*radius] >= cval ? 8   : 0) |
                                        (p[(r+1*radius)*m.cols+c+0*radius] >= cval ? 4   : 0) |
                                        (p[(r+1*radius)*m.cols+c-1*radius] >= cval ? 2   : 0) |
                                        (p[(r+0*radius)*m.cols+c-1*radius] >= cval ? 1   : 0)];
            }
        }

        dst += n;
    }
};

BR_REGISTER(Transform, LBPTransform)

} // namespace br

#include "imgproc/lbp.moc"
