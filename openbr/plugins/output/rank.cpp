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
#include <openbr/core/common.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Outputs highest ranked matches with scores.
 * \author Scott Klum \cite sklum
 */
class rankOutput : public MatrixOutput
{
    Q_OBJECT

    ~rankOutput()
    {
        if (targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        QList<int> ranks;
        QList<int> positions;
        QList<float> scores;
        QStringList lines;

        for (int i=0; i<queryFiles.size(); i++) {
            typedef QPair<float,int> Pair;
            int rank = 1;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(data.row(i)), true)) {
                // Check if target files are marked as allParitions, and make sure target and query files are in the same partition
                if (Globals->crossValidate > 0 ? (targetFiles[pair.second].get<int>("Partition",-1) == -1 || targetFiles[pair.second].get<int>("Partition",-1) == queryFiles[i].get<int>("Partition",-1)) : true) {
                    if (QString(targetFiles[pair.second]) != QString(queryFiles[i])) {
                        if (targetFiles[pair.second].get<QString>("Label") == queryFiles[i].get<QString>("Label")) {
                            ranks.append(rank);
                            positions.append(pair.second);
                            scores.append(pair.first);
                            break;
                        }
                        rank++;
                    }
                }
            }
        }

        typedef QPair<int,int> RankPair;
        foreach (const RankPair &pair, Common::Sort(ranks, false))
            // pair.first == rank retrieved, pair.second == original position
            lines.append(queryFiles[pair.second].name + " " + QString::number(pair.first) + " " + QString::number(scores[pair.second]) + " " + targetFiles[positions[pair.second]].name);


        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, rankOutput)

} // namespace br

#include "output/rank.moc"
