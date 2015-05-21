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
 * \brief Rank retrieval output.
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 */
class rrOutput : public MatrixOutput
{
    Q_OBJECT

    ~rrOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        const int limit = file.get<int>("limit", 20);
        const bool byLine = file.getBool("byLine");
        const bool simple = file.getBool("simple");
        const float threshold = file.get<float>("threshold", -std::numeric_limits<float>::max());

        QStringList lines;

        for (int i=0; i<queryFiles.size(); i++) {
            QStringList files;
            if (simple) files.append(queryFiles[i].fileName());

            typedef QPair<float,int> Pair;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(data.row(i)), true, limit)) {
                // Check if target files are marked as allParitions, and make sure target and query files are in the same partition
                if (Globals->crossValidate > 0 ? (targetFiles[pair.second].get<int>("Partition",-1) == -1 || targetFiles[pair.second].get<int>("Partition",-1) == queryFiles[i].get<int>("Partition",-1)) : true) {
                    if (pair.first < threshold) break;
                    File target = targetFiles[pair.second];
                    target.set("Score", QString::number(pair.first));
                    if (simple) files.append(target.fileName() + " " + QString::number(pair.first));
                    else files.append(target.flat());
                }
            }
            lines.append(files.join(byLine ? "\n" : ","));
        }

        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, rrOutput)

} // namespace br

#include "output/rr.moc"
