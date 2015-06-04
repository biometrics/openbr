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
 * \brief Score histogram.
 * \author Josh Klontz \cite jklontz
 */
class histOutput : public Output
{
    Q_OBJECT

    float min, max, step;
    QVector<int> bins;

    ~histOutput()
    {
        if (file.isNull() || bins.isEmpty()) return;
        QStringList counts;
        foreach (int count, bins)
            counts.append(QString::number(count));
        const QString result = counts.join(",");
        QtUtils::writeFile(file, result);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        min = file.get<float>("min", -5);
        max = file.get<float>("max", 5);
        step = file.get<float>("step", 0.1);
        bins = QVector<int>((max-min)/step, 0);
    }

    void set(float value, int i, int j)
    {
        (void) i;
        (void) j;
        if ((value < min) || (value >= max)) return;
        bins[(value-min)/step]++; // This should technically be locked to ensure atomic increment
    }
};

BR_REGISTER(Output, histOutput)

} // namespace br

#include "output/hist.moc"
