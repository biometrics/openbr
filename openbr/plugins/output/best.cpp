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
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief The highest scoring matches.
 * \author Josh Klontz \cite jklontz
 */
class bestOutput : public Output
{
    Q_OBJECT

    typedef QPair< float, QPair<int, int> > BestMatch;
    QList<BestMatch> bestMatches;

    ~bestOutput()
    {
        if (file.isNull() || bestMatches.isEmpty()) return;
        qSort(bestMatches);
        QStringList lines; lines.reserve(bestMatches.size()+1);
        lines.append("Value,Target,Query");
        for (int i=bestMatches.size()-1; i>=0; i--)
            lines.append(QString::number(bestMatches[i].first) + "," + targetFiles[bestMatches[i].second.second] + "," + queryFiles[bestMatches[i].second.first]);
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        bestMatches.reserve(queryFiles.size());
        for (int i=0; i<queryFiles.size(); i++)
            bestMatches.append(BestMatch(-std::numeric_limits<float>::max(), QPair<int,int>(-1, -1)));
    }

    void set(float value, int i, int j)
    {
        static QMutex lock;

        // Return early for self similar matrices
        if (selfSimilar && (i == j)) return;

        if (value > bestMatches[i].first) {
            lock.lock();
            if (value > bestMatches[i].first)
                bestMatches[i] = BestMatch(value, QPair<int,int>(i,j));
            lock.unlock();
        }
    }
};

BR_REGISTER(Output, bestOutput)

} // namespace br

#include "output/best.moc"
