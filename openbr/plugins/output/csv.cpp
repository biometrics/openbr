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
 * \brief Comma separated values output.
 * \author Josh Klontz \cite jklontz
 */
class csvOutput : public MatrixOutput
{
    Q_OBJECT

    ~csvOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QStringList lines;
        lines.append("File," + targetFiles.names().join(","));
        for (int i=0; i<queryFiles.size(); i++) {
            QStringList words;
            for (int j=0; j<targetFiles.size(); j++)
                words.append(toString(i,j));  // The toString idiom is used to output match scores - see MatrixOutput
            lines.append(queryFiles[i].name+","+words.join(","));
        }
        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, csvOutput)

} // namespace br

#include "output/csv.moc"
