
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
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
 * \ingroup transforms
 * \brief Performs majority voting from a single metadata key or multiple keys and sets the result in the specified key.
 * \br_property QString keys The meta-data key(s) used for Majority Voting.
 * \br_property int numClasses The number of possible classes.
 * \br_property QString outputKey The output metadata key which stores the a list with the index of the winning class set to 1.
 * \br_property float thresh Allows users to specify a threshold for the first class, if exceeded the first class wins else class 2 wins.
 * \author Keyur Patel \cite kpatel
 */
class MajorityVotingTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    Q_PROPERTY(int numClasses READ get_numClasses WRITE set_numClasses RESET reset_numClasses STORED false)
    Q_PROPERTY(QString outputKey READ get_outputKey WRITE set_outputKey RESET reset_outputKey STORED false)
    Q_PROPERTY(float thresh READ get_thresh WRITE set_thresh RESET reset_thresh STORED false)

    BR_PROPERTY(QStringList, keys, QStringList())
    BR_PROPERTY(int, numClasses, 2)
    BR_PROPERTY(QString, outputKey, "MajorityVoting")
    BR_PROPERTY(float, thresh, -1)  

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        QList<float> scores = QList<float>();
        foreach (QString key, keys){
            if (src.contains(key)){
                QList<float> templateScores = src.getList<float>(key);
                scores.append(templateScores);
            } else {
                dst.fte = true;
                return;
            }
        }
        QVector<int> classCount(numClasses, 0);
        for (int c = 0; c < scores.size(); c+= numClasses){
            if ((numClasses == 2) && (thresh != -1)){
                if (scores[c] > thresh)
                    classCount[0]++;
                else
                    classCount[1]++;
            } else {
                int highestScoringClass = 0;
                float highestScore = scores[c];
                for (int b = 1; b < numClasses; b++){
                    if (scores[c+b] >  highestScore){
                        highestScore = scores[c+b];
                        highestScoringClass = b;
                    }
                }
                classCount[highestScoringClass]++;
            }
        }
        int largestIndex = getIndexOfLargestElement(classCount);

        QList<int> output = QList<int>();
        for(int i=0; i <numClasses; i++){
            if(i == largestIndex) output.append(1);
            else output.append(0);
        }
        dst.setList(outputKey,output);
    }

    int getIndexOfLargestElement(const QVector<int> &arr) const {
        int largestIndex = 0;
        for (int i = largestIndex; i < arr.size(); i++) {
            if (arr[largestIndex] <= arr[i]) {
                largestIndex = i;
            }
        }
        return largestIndex;
    }
};

BR_REGISTER(Transform, MajorityVotingTransform)

} // namespace br

#include "metadata/majorityvoting.moc"

