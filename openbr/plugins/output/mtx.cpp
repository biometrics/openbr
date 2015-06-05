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
 * \brief simmat output.
 * \author Josh Klontz \cite jklontz
 */
class mtxOutput : public Output
{
    Q_OBJECT

    Q_PROPERTY(QString targetGallery READ get_targetGallery WRITE set_targetGallery RESET reset_targetGallery STORED false)
    Q_PROPERTY(QString queryGallery READ get_queryGallery WRITE set_queryGallery RESET reset_queryGallery STORED false)
    BR_PROPERTY(QString, targetGallery, "Unknown_Target")
    BR_PROPERTY(QString, queryGallery, "Unknown_Query")

    int headerSize, rowBlock, columnBlock;
    cv::Mat blockScores;

    ~mtxOutput()
    {
        writeBlock();
    }

    void setBlock(int rowBlock, int columnBlock)
    {
        if ((rowBlock == 0) && (columnBlock == 0)) {
            // Initialize the file
            QFile f(file);
            QtUtils::touchDir(f);
            if (!f.open(QFile::WriteOnly))
                qFatal("Unable to open %s for writing.", qPrintable(file));
            const int endian = 0x12345678;
            QByteArray header;
            header.append("S2\n");
            header.append(qPrintable(targetGallery));
            header.append("\n");
            header.append(qPrintable(queryGallery));
            header.append("\nMF ");
            header.append(qPrintable(QString::number(queryFiles.size())));
            header.append(" ");
            header.append(qPrintable(QString::number(targetFiles.size())));
            header.append(" ");
            header.append(QByteArray((const char*)&endian, 4));
            header.append("\n");
            headerSize = f.write(header);
            const float defaultValue = -std::numeric_limits<float>::max();
            for (int i=0; i<targetFiles.size()*queryFiles.size(); i++)
                f.write((const char*)&defaultValue, 4);
            f.close();
        } else {
            writeBlock();
        }

        this->rowBlock = rowBlock;
        this->columnBlock = columnBlock;

        int matrixRows  = std::min(queryFiles.size()-rowBlock*this->blockRows, blockRows);
        int matrixCols  = std::min(targetFiles.size()-columnBlock*this->blockCols, blockCols);

        blockScores = cv::Mat(matrixRows, matrixCols, CV_32FC1);
    }

    void setRelative(float value, int i, int j)
    {
        blockScores.at<float>(i,j) = value;
    }

    void set(float value, int i, int j)
    {
        (void) value; (void) i; (void) j;
        qFatal("Logic error.");
    }

    void writeBlock()
    {
        QFile f(file);
        if (!f.open(QFile::ReadWrite))
            qFatal("Unable to open %s for modifying.", qPrintable(file));
        for (int i=0; i<blockScores.rows; i++) {
            f.seek(headerSize + sizeof(float)*(quint64(rowBlock*this->blockRows+i)*targetFiles.size()+(columnBlock*this->blockCols)));
            f.write((const char*)blockScores.row(i).data, sizeof(float)*blockScores.cols);
        }
        f.close();
    }
};

BR_REGISTER(Output, mtxOutput)

} // namespace br

#include "output/mtx.moc"
