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
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Weka ARFF file format.
 * \author Josh Klontz \cite jklontz
 * \br_link http://weka.wikispaces.com/ARFF+%28stable+version%29
 */
class arffGallery : public Gallery
{
    Q_OBJECT
    QFile arffFile;

    TemplateList readBlock(bool *done)
    {
        (void) done;
        qFatal("Not implemented.");
        return TemplateList();
    }

    void write(const Template &t)
    {
        if (!arffFile.isOpen()) {
            arffFile.setFileName(file.name);
            arffFile.open(QFile::WriteOnly);
            arffFile.write("% OpenBR templates\n"
                           "@RELATION OpenBR\n"
                           "\n");

            const int dimensions = t.m().rows * t.m().cols;
            for (int i=0; i<dimensions; i++)
                arffFile.write(qPrintable("@ATTRIBUTE v" + QString::number(i) + " REAL\n"));
            arffFile.write(qPrintable("@ATTRIBUTE class string\n"));

            arffFile.write("\n@DATA\n");
        }

        arffFile.write(qPrintable(OpenCVUtils::matrixToStringList(t).join(',')));
        arffFile.write(qPrintable(",'" + t.file.get<QString>("Label") + "'\n"));
    }
};

BR_REGISTER(Gallery, arffGallery)

} // namespace br

#include "gallery/arff.moc"
