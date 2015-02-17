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
class tailOutput : public Output
{
    Q_OBJECT

    struct Comparison
    {
        br::File query, target;
        float value;

        Comparison(const br::File &_query, const br::File &_target, float _value)
            : query(_query), target(_target), value(_value) {}

        QString toString(bool args) const
        {
            return QString::number(value) + "," + (args ? target.flat() : (QString)target) + "," + (args ? query.flat() : (QString)query);
        }

        bool operator<(const Comparison &other) const
        {
            return value < other.value;
        }
    };

    float threshold;
    int atLeast, atMost;
    bool args;
    float lastValue;
    QList<Comparison> comparisons;
    QMutex comparisonsLock;

    ~tailOutput()
    {
        if (file.isNull() || comparisons.isEmpty()) return;
        QStringList lines; lines.reserve(comparisons.size()+1);
        lines.append("Value,Target,Query");
        foreach (const Comparison &duplicate, comparisons)
            lines.append(duplicate.toString(args));
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        threshold = file.get<float>("threshold", -std::numeric_limits<float>::max());
        atLeast = file.get<int>("atLeast", 1);
        atMost = file.get<int>("atMost", std::numeric_limits<int>::max());
        args = file.get<bool>("args", false);
        lastValue = -std::numeric_limits<float>::max();
    }

    void set(float value, int i, int j)
    {
        // Return early for self similar matrices
        if (selfSimilar && (i <= j)) return;

        // Consider only values passing the criteria
        if ((value < threshold) && (value <= lastValue) && (comparisons.size() >= atLeast))
            return;

        comparisonsLock.lock();
        if (comparisons.isEmpty() || (value < comparisons.last().value)) {
            // Special cases
            comparisons.append(Comparison(queryFiles[i], targetFiles[j], value));
        } else {
            // General case
            for (int k=0; k<comparisons.size(); k++) {
                if (comparisons[k].value <= value) {
                    comparisons.insert(k, Comparison(queryFiles[i], targetFiles[j], value));
                    break;
                }
            }
        }

        while (comparisons.size() > atMost)
            comparisons.removeLast();
        while ((comparisons.size() > atLeast) && (comparisons.last().value < threshold))
            comparisons.removeLast();

        lastValue = comparisons.last().value;
        comparisonsLock.unlock();
    }
};

BR_REGISTER(Output, tailOutput)

} // namespace br

#include "output/tail.moc"
