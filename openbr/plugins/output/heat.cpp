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
 * \brief Matrix-like output for heat maps.
 * \author Scott Klum \cite sklum
 */
class heatOutput : public MatrixOutput
{
    Q_OBJECT
    Q_PROPERTY(int patches READ get_patches WRITE set_patches RESET reset_patches STORED false)
    BR_PROPERTY(int, patches, -1)

    ~heatOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        QStringList lines;
        for (int i=0; i<data.rows; i++) {
            lines.append(toString(i,0));
        }
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        if (patches == -1) qFatal("Heat output requires the number of patches");
        Output::initialize(targetFiles, queryFiles);
        data.create(patches, 1, CV_32FC1);
    }
};

BR_REGISTER(Output, heatOutput)

} // namespace br

#include "output/heat.moc"
