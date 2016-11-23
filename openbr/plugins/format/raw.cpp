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
 * \brief RAW format
 *
 * \author Scott Klum \cite sklum
 */
class rawFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        QByteArray data;
        QtUtils::readFile(file, data);

        // The raw file format has no header information, so one must specify resolution
        QSize size = QSize(file.get<int>("width"),file.get<int>("height"));
        Template t(file);
        QList<Mat> matrices;
        const int bytes = size.width()*size.height();
        for (int i=0; i<data.size()/(size.height()*size.width()); i++)
            t.append(Mat(size.height(), size.width(), CV_8UC1, data.data()+bytes*i).clone());
        return t;
    }

    void write(const Template &t) const
    {
        QtUtils::writeFile(file, QByteArray().setRawData((const char*)t.m().data, t.m().total() * t.m().elemSize()));
    }
};

BR_REGISTER(Format, rawFormat)

} // namespace br

#include "format/raw.moc"
