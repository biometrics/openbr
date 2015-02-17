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
 * \ingroup outputs
 * \brief Output to the terminal.
 * \author Josh Klontz \cite jklontz
 */
class EmptyOutput : public MatrixOutput
{
    Q_OBJECT

    static QString bufferString(const QString &string, int length)
    {
        if (string.size() >= length)
            return string.left(length);
        QString buffer; buffer.fill(' ', length-string.size());
        return string+buffer;
    }

    ~EmptyOutput()
    {
        if (targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QString result;
        if ((queryFiles.size() == 1) && (targetFiles.size() == 1)) {
            result = toString(0,0) + "\n";
        } else {
            const int CELL_SIZE = 12;

            result = bufferString(" ", CELL_SIZE) + " ";
            foreach (const QString &targetName, targetFiles.names())
                result += bufferString(targetName, CELL_SIZE) + " ";
            result += "\n";

            for (int i=0; i<queryFiles.size(); i++) {
                result += bufferString(queryFiles[i].name, CELL_SIZE) + " ";
                for (int j=0; j<targetFiles.size(); j++)
                    result += bufferString(toString(i,j), CELL_SIZE) + " ";
                result += "\n";
            }
        }

        printf("%s", qPrintable(result));
    }
};

BR_REGISTER(Output, EmptyOutput)

} // namespace br

#include "output/empty.moc"
