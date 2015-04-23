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
    bool create( const std::string _posFilename, const std::string _negFilename, cv::Size _winSize );
    void restart() { posReader.restart(); }
    bool getNeg(cv::Mat &_img) { return negReader.get( _img ); }
    bool getPos(cv::Mat &_img) { return posReader.get( _img ); }

private:
    class PosReader
    {
    public:
        PosReader();
        virtual ~PosReader();
        bool create( const std::string _filename );
        bool get( cv::Mat &_img );
        void restart();

        short* vec;
        FILE*  file;
        int    count;
        int    vecSize;
        int    last;
        int    base;
    } posReader;

    class NegReader
    {
    public:
        NegReader();
        bool create( const std::string _filename, cv::Size _winSize );
        bool get( cv::Mat& _img );
        bool nextImg();

        cv::Mat     src, img;
        std::vector<std::string> imgFilenames;
        cv::Point   offset, point;
        float   scale;
        float   scaleFactor;
        float   stepFactor;
        size_t  last, round;
        cv::Size    winSize;
    } negReader;
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
                const std::string _posFilename,
                const std::string _negFilename,
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

#endif // CASCADE_H
