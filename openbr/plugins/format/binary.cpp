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
 * \brief A simple binary matrix format.
 * \author Josh Klontz \cite jklontz
 * \br_format First 4 bytes indicate the number of rows.
 * Second 4 bytes indicate the number of columns.
 * The rest of the bytes are 32-bit floating data elements in row-major order.
 */
class binaryFormat : public Format
{
    Q_OBJECT
    Q_PROPERTY(bool raw READ get_raw WRITE set_raw RESET reset_raw STORED false)
    BR_PROPERTY(bool, raw, false)

    Template read() const
    {
        QByteArray data;
        QtUtils::readFile(file, data);
        if (raw) {
            return Template(file, Mat(1, data.size(), CV_8UC1, data.data()).clone());
        } else {
            return Template(file, Mat(((quint32*)data.constData())[0],
                                      ((quint32*)data.constData())[1],
                                      CV_32FC1,
                                      data.data()+8).clone());
        }
    }

    void write(const Template &t) const
    {
        QFile f(file);
        QtUtils::touchDir(f);
        if (!f.open(QFile::WriteOnly))
            qFatal("Failed to open %s for writing.", qPrintable(file));

        Mat m;
        if (!raw) {
            if (t.m().type() != CV_32FC1)
                t.m().convertTo(m, CV_32F);
            else m = t.m();

            if (m.channels() != 1) qFatal("Only supports single channel matrices.");

            f.write((const char *) &m.rows, 4);
            f.write((const char *) &m.cols, 4);
        }
        else m =  t.m();

        qint64 rowSize = m.cols * sizeof(float);
        for (int i=0; i < m.rows; i++)
        {
            f.write((const char *) m.row(i).data, rowSize);
        }
        f.close();
    }
};

BR_REGISTER(Format, binaryFormat)

} // namespace br

#include "format/binary.moc"
