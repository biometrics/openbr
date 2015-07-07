/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Corporation                                       *
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
 * \ingroup galleries
 * \brief A simple matrix container format that is easily read into matlab. 
 *          Templates are concatenated into column vectors, and output into a single matrix.
 *          Note that this is not intended to read in .matlab files, as this is simply 
 *          a quick and dirty for analyzing data in a more interactive environment.
 *          Use the loadOpenBR.m script to ingest the resultant file into Matlab
 * \br_property bool writeLabels Write the subject labels of the instances in the final row of the matrix.
 * \author Brendan Klare \cite bklare
 */
class matlabGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(bool writeLabels READ get_writeLabels WRITE set_writeLabels RESET reset_writeLabels STORED false)
    BR_PROPERTY(bool, writeLabels, false)

    TemplateList templates;

    ~matlabGallery()
    {
        if (!f.open(QFile::WriteOnly))
            qFatal("Failed to open %s for writting.", qPrintable(file));

        int r = templates.first().m().rows;     
        int c = templates.first().m().cols;     

        for (int i = templates.size() - 1; i >= 0; i--) {        
            if (templates[i].m().rows != r || templates[i].m().cols != c) {
                qDebug() << templates[i].file.name << " : template not appended to gallery b/c rows and col dimensions differ.";
                templates.removeAt(i);
            }
        }

        cv::Mat m(r * c, templates.size(), CV_32FC1);
        for (int i = 0; i < templates.size(); i++) {
            cv::Mat temp;
            temp = templates[i].m().reshape(1, r * c);
            temp.copyTo(m.col(i));
        }

        if (writeLabels) {
            TemplateList _templates = TemplateList::relabel(templates, "Label", false);

            QList<int> classes = File::get<int>(_templates, "Label");
            cv::Mat _m(r * c + 1, _templates.size(), CV_32FC1);
            m.copyTo(_m.rowRange(cv::Range(0, r * c)));
            for (int i = 0; i < _templates.size(); i++) {
                _m.at<float>(r * c, i) = (float) classes[i];
            }
            m = _m;
        }

        f.write((const char *) &m.rows, 4);
        f.write((const char *) &m.cols, 4);
        qint64 rowSize = m.cols * sizeof(float);
        for (int i = 0; i < m.rows; i++) 
            f.write((const char *) m.row(i).data, rowSize);

        f.close();
    }

    TemplateList readBlock(bool *done)
    {
        qDebug() << "matlabGallery is not intended to be read. This gallery is meant for export only.";
        TemplateList t;
        *done = false;
        return t;
    }

    void write(const Template &t)
    {
        if (t.m().type() != CV_32FC1) 
            qDebug() << t.file.name << ": template not appended to gallery b/c it must be type CV_32FC1.";
        else
            templates.append(t);
    }

    void init()
    {
        FileGallery::init();
    }
};

BR_REGISTER(Gallery, matlabGallery)

}

#include "gallery/matlab.moc"
