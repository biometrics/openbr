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

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads a NIST LFFS file.
 * \author Josh Klontz \cite jklontz
 */
class lffsFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        QByteArray byteArray;
        QtUtils::readFile(file.name, byteArray);
        return Mat(1, byteArray.size(), CV_8UC1, byteArray.data()).clone();
    }

    void write(const Template &t) const
    {
        QByteArray byteArray((const char*)t.m().data, t.m().total()*t.m().elemSize());
        QtUtils::writeFile(file.name, byteArray);
    }
};

BR_REGISTER(Format, lffsFormat)

} // namespace br

#include "format/lffs.moc"
