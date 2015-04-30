#ifndef CASCADE_H
#define CASCADE_H

#include <openbr/openbr_plugin.h>
#include <opencv2/highgui/highgui.hpp>
#include "features.h"
#include "boost.h"

namespace br
{

class CascadeImageReader
{
public:
    bool create( const QList<cv::Mat> &_posImages, const QList<cv::Mat> &_negImages, cv::Size _winSize );
    void restart() { posIdx = 0; }
    bool getNeg(cv::Mat &_img);
    bool getPos(cv::Mat &_img);

    QList<cv::Mat> posImages, negImages;

    int posIdx, negIdx;

    cv::Mat     src, img;
    cv::Point   offset, point;
    float   scale;
    float   scaleFactor;
    float   stepFactor;
    size_t  round;
    cv::Size    winSize;

private:
    bool nextNeg();
};


class BrCascadeClassifier
{
public:
    bool train(const std::string _cascadeDirName,
               const QList<cv::Mat> &_posImages,
               const QList<cv::Mat> &_negImages,
               int _numPos, int _numNeg,
               int _precalcValBufSize, int _precalcIdxBufSize,
               int _numStages,
               cv::Size _winSize,
               const CascadeBoostParams& _stageParams);
private:
    int predict(int sampleIdx);
    void save(const std::string cascadeDirName);
    bool updateTrainingSet(double& acceptanceRatio);
    int fillPassedSamples(int first, int count, bool isPositive, int64& consumed);

    void writeParams(cv::FileStorage &fs) const;
    void writeStages(cv::FileStorage &fs, const cv::Mat& featureMap) const;
    void writeFeatures(cv::FileStorage &fs, const cv::Mat& featureMap) const;

    void getUsedFeaturesIdxMap(cv::Mat& featureMap);

    cv::Ptr<CascadeBoostParams> stageParams;

    cv::Ptr<FeatureEvaluator> featureEvaluator;
    std::vector< cv::Ptr<CascadeBoost> > stageClassifiers;
    CascadeImageReader imgReader;
    int numStages, curNumSamples;
    int numPos, numNeg;
    cv::Size winSize;
};

} // namespace br

#endif // CASCADE_H
