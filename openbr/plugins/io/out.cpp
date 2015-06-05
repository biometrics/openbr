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
 * \brief DOCUMENT ME CHARLES
 * \author Unknown \cite Unknown
 */
class OutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputString READ get_outputString WRITE set_outputString RESET reset_outputString STORED false)
    // names of mem galleries containing filelists we need.
    Q_PROPERTY(QString targetName READ get_targetName WRITE set_targetName RESET reset_targetName STORED false)
    Q_PROPERTY(QString queryName  READ get_queryName WRITE set_queryName RESET reset_queryName STORED false)
    Q_PROPERTY(bool transposeMode  READ get_transposeMode WRITE set_transposeMode RESET reset_transposeMode STORED false)

    BR_PROPERTY(QString, outputString, "")
    BR_PROPERTY(QString, targetName, "")
    BR_PROPERTY(QString, queryName, "")
    BR_PROPERTY(bool, transposeMode, false)

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (src.empty())
            return;

        // we received a template, which is the next row/column in order
        foreach (const Template &t, dst) {
            bool fte = t.file.getBool("FTE") || t.file.fte;

            for (int i=0; i < scoresPerMat; i++) {
                output->setRelative(fte ? -std::numeric_limits<float>::max() : t.m().at<float>(0, i), currentRow, currentCol);

                // row-major input
                if (!transposeMode)
                    currentCol++;
                // col-major input
                else
                    currentRow++;
            }
            // filled in a row, advance to the next, reset column position
            if (!transposeMode) {
                currentRow++;
                currentCol = 0;
            }
            // filled in a column, advance, reset row
            else {
                currentCol++;
                currentRow = 0;
            }

            bool blockDone = false;
            // In direct mode, we don't buffer rows
            if (!transposeMode) {
                currentBlockRow++;
                blockDone = true;
            }
            // in transpose mode, we buffer 100 cols before writing the block
            else if (currentCol == bufferedSize) {
                currentBlockCol++;
                blockDone = true;
            }
            else return;

            if (blockDone) {
                // set the next block, only necessary if we haven't buffered the current item
                output->setBlock(currentBlockRow, currentBlockCol);
                currentRow = 0;
                currentCol = 0;
            }
        }
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    void init()
    {
        if (targetName.isEmpty() || queryName.isEmpty() || outputString.isEmpty())
            return;

        FileList targetFiles = FileList::fromGallery(targetName);
        FileList queryFiles  = FileList::fromGallery(queryName);

        currentBlockRow = 0;
        currentBlockCol = 0;

        currentRow = 0;
        currentCol = 0;

        bufferedSize = 100;

        if (transposeMode) {
            // buffer 100 cols at a time
            fragmentsPerRow = bufferedSize;
            // a single col contains comparisons to all query files
            fragmentsPerCol = queryFiles.size();
            scoresPerMat = fragmentsPerCol;
        }
        else {
            // a single row contains comparisons to all target files
            fragmentsPerRow = targetFiles.size();
            scoresPerMat = fragmentsPerRow;
            // we output rows one at a time
            fragmentsPerCol = 1;
        }

        output = QSharedPointer<Output>(Output::make(outputString+"[targetGallery="+targetName+",queryGallery="+queryName+"]", targetFiles, queryFiles));
        output->blockRows = fragmentsPerCol;
        output->blockCols = fragmentsPerRow;
        output->initialize(targetFiles, queryFiles);

        output->setBlock(currentBlockRow, currentBlockCol);
    }

    QSharedPointer<Output> output;

    int bufferedSize;

    int currentRow;
    int currentCol;

    int currentBlockRow;
    int currentBlockCol;

    int fragmentsPerRow;
    int fragmentsPerCol;

    int scoresPerMat;

public:
    OutputTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, OutputTransform)

} // namespace br

#include "io/out.moc"
