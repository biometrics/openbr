#include "features.h"

using namespace cv;
using namespace br;

//------------------------------------- FeatureEvaluator ---------------------------------------

void FeatureEvaluator::init(Representation *_representation, int _maxSampleCount)
{
    representation = _representation;
    data.create((int)_maxSampleCount, representation->postWindowSize().area(), CV_32SC1);
    cls.create( (int)_maxSampleCount, 1, CV_32FC1 );
}

void FeatureEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    cls.ptr<float>(idx)[0] = clsLabel;

    Mat integralImg(representation->postWindowSize(), data.type(), data.ptr<int>(idx));
    representation->preprocess(img, integralImg);
}

void FeatureEvaluator::writeFeatures(FileStorage &fs, const Mat &featureMap) const
{
    representation->write(fs, featureMap);
}
