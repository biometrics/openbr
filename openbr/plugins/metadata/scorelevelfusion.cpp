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
#include <opencv2/core/core.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Fuses together scores from a single metadata key or multiple keys and sets the computed values in the specified key.
 * \br_property QString keys The meta-data key(s) to fuse on.
 * \br_property QString outputKey The output metadata key.
 * \br_property bool listOfList Flag that tells the program how to parse the metadata key that contains the score values.
 * \author Keyur Patel \cite kpatel
 */
class ScoreLevelFusionTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    Q_PROPERTY(int numClasses READ get_numClasses WRITE set_numClasses RESET reset_numClasses STORED false)
    Q_PROPERTY(QString outputKey READ get_outputKey WRITE set_outputKey RESET reset_outputKey STORED false)
    Q_PROPERTY(bool listOfList READ get_listOfList WRITE set_listOfList RESET reset_listOfList STORED false)

    BR_PROPERTY(QStringList, keys, QStringList())
    BR_PROPERTY(int, numClasses, 2)
    BR_PROPERTY(QString, outputKey, "ScoreLevelFusion")
    BR_PROPERTY(bool, listOfList, false)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
	QList<QList<float> > scoresList = QList<QList<float> >();
	if (listOfList){ //format [[.8,.2],[.45,.55],[.3,.7]]
            foreach (QString key, keys){
            	if (src.contains(key)){
                    QList<QString > keyScoresList = src.getList<QString >(key);
                    for (int d = 0; d < keyScoresList.size(); d++){
                   	QList<QString> tmpList = keyScoresList[d].split(",");
                    	QList<float> tmpScoreList = QList<float>();
                    	for (int f = 0; f < tmpList.size(); f++){
                            tmpScoreList.append(tmpList[f].remove('"').remove("]").remove("[").toFloat());
                        }
                        scoresList.append(tmpScoreList);
                    }
                } else{
                    dst.fte = true;
                    return;
                }
            }
	} else {  //format = [.8,.2,.45,.55,.3.,7]
	    QList<float> scores = QList<float>();
            foreach (QString key, keys){
            	if (src.contains(key)){
                    QList<float> templateScores = src.getList<float>(key);
                    scores.append(templateScores);
            	} else{
                    dst.fte = true;
                    return;
            	}
            }
	    for (int i = 0; i < scores.size(); i+=numClasses){
		QList<float> tmpList = QList<float>();
		for (int b = 0; b < numClasses; b++){
		    tmpList.append(scores[i+b]);
		}
		scoresList.append(tmpList);
	    }
	}
        Mat m = toMat(scoresList);
        Mat avgDist;
        cv::reduce(m, avgDist, 0, 1);
        dst.setList(outputKey,matrixToVector(avgDist));
    }

    cv::Mat toMat(const QList<QList<float> > &src) const
        {
            cv::Mat dst(src.size(), src[0].size(), CV_32F);
            for (int i=0; i<src.size(); i++)
                for (int j=0; j<src[i].size(); j++)
                    dst.at<float>(i, j) = src[i][j];
            return dst;
        }

    QList<float> matrixToVector(const cv::Mat &m) const
    {
        QList<float> results;
        for (int i=0; i<m.cols; i++)
            results.append(m.at<float>(0, i));
        return results;
    }

};

BR_REGISTER(Transform, ScoreLevelFusionTransform)

} // namespace br

#include "metadata/scorelevelfusion.moc"

