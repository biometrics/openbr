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

#include <fstream>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Log nearest neighbors to specified file.
 * \author Charles Otto \cite caotto
 * \br_property QString fileName The name of the log file. An empty fileName won't be written to. Default is "".
 */
class LogNNTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString fileName READ get_fileName WRITE set_fileName RESET reset_fileName STORED false)
    BR_PROPERTY(QString, fileName, "")

    std::fstream fout;

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;

        if (!dst.file.contains("neighbors")) {
            fout << std::endl;
            return;
        }

        Neighbors neighbors = dst.file.get<Neighbors>("neighbors");
        if (neighbors.isEmpty() ) {
            fout << std::endl;
            return;
        }

        QString aLine;
        aLine.append(QString::number(neighbors[0].first)+":"+QString::number(neighbors[0].second));
        for (int i=1; i < neighbors.size();i++)
            aLine.append(","+QString::number(neighbors[i].first)+":"+QString::number(neighbors[i].second));

        fout << qPrintable(aLine) << std::endl;
    }

    void init()
    {
        if (!fileName.isEmpty())
            fout.open(qPrintable(fileName), std::ios_base::out);
    }

    void finalize(TemplateList &output)
    {
        (void) output;
        fout.close();
    }

public:
    LogNNTransform() : TimeVaryingTransform(false, false) {}
};

BR_REGISTER(Transform, LogNNTransform)

} // namespace br

#include "cluster/lognn.moc"
