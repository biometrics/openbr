#include "features.h"
#include "opencvutils.h"

using namespace cv;
using namespace br;

// ------------------------------------ LBP Training -----------------------------------------------

void LBPTrainingEvaluator::init(int _maxSampleCount, Size _winSize)
{
    CV_Assert( _maxSampleCount > 0);
    sum.create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32SC1);

    winSize = _winSize;
    numFeatures = 0;
    maxCatCount = 256;
    cls.create( (int)_maxSampleCount, 1, CV_32FC1 );
    generateFeatures();
}

void LBPTrainingEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !sum.empty() );
    cls.ptr<float>(idx)[0] = clsLabel;
    Mat innSum(winSize.height + 1, winSize.width + 1, sum.type(), sum.ptr<int>((int)idx));
    integral( img, innSum );
}

void LBPTrainingEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    _writeFeatures( features, fs, featureMap );
}

void LBPTrainingEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    for( int x = 0; x < winSize.width; x++ )
        for( int y = 0; y < winSize.height; y++ )
            for( int w = 1; w <= winSize.width / 3; w++ )
                for( int h = 1; h <= winSize.height / 3; h++ )
                    if ( (x+3*w <= winSize.width) && (y+3*h <= winSize.height) )
                        features.push_back( Feature(offset, x, y, w, h ) );
    numFeatures = (int)features.size();
}

LBPTrainingEvaluator::Feature::Feature()
{
    rect = cvRect(0, 0, 0, 0);
}

LBPTrainingEvaluator::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight )
{
    Rect tr = rect = cvRect(x, y, _blockWidth, _blockHeight);
    CV_SUM_OFFSETS( p[0], p[1], p[4], p[5], tr, offset )
    tr.x += 2*rect.width;
    CV_SUM_OFFSETS( p[2], p[3], p[6], p[7], tr, offset )
    tr.y +=2*rect.height;
    CV_SUM_OFFSETS( p[10], p[11], p[14], p[15], tr, offset )
    tr.x -= 2*rect.width;
    CV_SUM_OFFSETS( p[8], p[9], p[12], p[13], tr, offset )
}

void LBPTrainingEvaluator::Feature::write(FileStorage &fs) const
{
    fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}
