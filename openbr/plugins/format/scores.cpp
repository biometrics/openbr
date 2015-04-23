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
#include <openbr/core/opencvutils.h>
#include <openbr/core/bee.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads in scores or ground truth from a text table.
 * \author Josh Klontz \cite jklontz
 * \br_format Example of the format:
 *
 * 2.2531514    FALSE   99990377    99990164
 * 2.2549822    TRUE    99990101    99990101
 */
class scoresFormat : public Format
{
    Q_OBJECT
    Q_PROPERTY(int column READ get_column WRITE set_column RESET reset_column STORED false)
    Q_PROPERTY(bool groundTruth READ get_groundTruth WRITE set_groundTruth RESET reset_groundTruth STORED false)
    Q_PROPERTY(QString delimiter READ get_delimiter WRITE set_delimiter RESET reset_delimiter STORED false)
    BR_PROPERTY(int, column, 0)
    BR_PROPERTY(bool, groundTruth, false)
    BR_PROPERTY(QString, delimiter, "\t")

    Template read() const
    {
        QFile f(file.name);
        if (!f.open(QFile::ReadOnly | QFile::Text))
            qFatal("Failed to open %s for reading.", qPrintable(f.fileName()));
        QList<float> values;
        while (!f.atEnd()) {
            const QStringList words = QString(f.readLine()).split(delimiter);
            if (words.size() <= column) qFatal("Expected file to have at least %d columns.", column+1);
            const QString &word = words[column];
            bool ok;
            float value = word.toFloat(&ok);
            if (!ok) value = (QtUtils::toBool(word) ? BEE::Match : BEE::NonMatch);
            values.append(value);
        }
        if (values.size() == 1)
            qWarning("Only one value read, double check file line endings.");
        Mat result = OpenCVUtils::toMat(values);
        if (groundTruth) result.convertTo(result, CV_8U);
        return result;
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Not implemented.");
    }
};

BR_REGISTER(Format, scoresFormat)

} // namespace br

#include "format/scores.moc"
