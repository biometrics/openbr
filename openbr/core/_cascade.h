#ifndef _CASCADE_H
#define _CASCADE_H

#include <openbr/openbr_plugin.h>
#include <opencv2/highgui/highgui.hpp>
#include "features.h"
#include "boost.h"

namespace br
{

class CascadeImageReader
{
public:
    bool create( const std::vector<cv::Mat> &_posImages, const std::vector<cv::Mat> &_negImages, cv::Size _winSize );
    void restart() { posIdx = 0; }
    bool getNeg(cv::Mat &_img);
    bool getPos(cv::Mat &_img);

private:
    std::vector<cv::Mat> posImages, negImages;

    int posIdx, negIdx;

    cv::Mat     src, img;
    cv::Point   offset, point;
    float   scale;
    float   scaleFactor;
    float   stepFactor;
    size_t  round;
    cv::Size    winSize;
};

class CascadeParams : public Params
{
public:
    enum { BOOST = 0 };
    static const int defaultStageType = BOOST;
    static const int defaultFeatureType = FeatureParams::LBP;

    CascadeParams();
    CascadeParams( int _stageType, int _featureType );
    void write( cv::FileStorage &fs ) const;
    bool read( const cv::FileNode &node );

    void printDefaults() const;
    void printAttrs() const;
    bool scanAttr( const std::string prmName, const std::string val );

    int stageType;
    int featureType;
    cv::Size winSize;
};

class BrCascadeClassifier
{
public:
    bool train( const std::string _cascadeDirName,
                const std::vector<cv::Mat> &_posImages,
                const std::vector<cv::Mat> &_negImages,
                int _numPos, int _numNeg,
                int _precalcValBufSize, int _precalcIdxBufSize,
                int _numStages,
                const CascadeParams& _cascadeParams,
                const FeatureParams& _featureParams,
                const CascadeBoostParams& _stageParams,
                bool baseFormatSave = false );
private:
    int predict( int sampleIdx );
    void save( const std::string cascadeDirName, bool baseFormat = false );
    bool load( const std::string cascadeDirName );
    bool updateTrainingSet( double& acceptanceRatio );
    int fillPassedSamples( int first, int count, bool isPositive, int64& consumed );

    void writeParams( cv::FileStorage &fs ) const;
    void writeStages( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    bool readParams( const cv::FileNode &node );
    bool readStages( const cv::FileNode &node );

    void getUsedFeaturesIdxMap( cv::Mat& featureMap );

    CascadeParams cascadeParams;
    cv::Ptr<FeatureParams> featureParams;
    cv::Ptr<CascadeBoostParams> stageParams;

    cv::Ptr<FeatureEvaluator> featureEvaluator;
    std::vector< cv::Ptr<CascadeBoost> > stageClassifiers;
    CascadeImageReader imgReader;
    int numStages, curNumSamples;
    int numPos, numNeg;
};

} // namespace br

#endif // _CASCADE_H

