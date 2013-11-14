#!/bin/bash

# Right now this is just a simple proof of concept. No quantitative eval is performed
# but instead the qualitative results are displayed.

# Make sure you set your data path. This will likely by your openbr/data directory.
if [ -z "$DATA" ]; then
    INRIA_PATH=../data/INRIAPerson
else
    INRIA_PATH=$DATA/INRIAPerson
fi

ALG="Open+Cvt(Gray)+Rename(neg,0)+BuildScales(Blur(2)+LBP(1,2)+SlidingWindow(Hist(59)+Cat+LDA(isBinary=true),windowWidth=10,takeLargestScale=false,threshold=2),windowWidth=10,takeLargestScale=false,minScale=4)+ConsolidateDetections+Discard"

# Josh's new algorithm (in progress)
# ALG2="Open+Cvt(Gray)+Detector(Gradient+Bin(0,360,9,true)+Merge+Integral+SlidingWindow(Identity))"

br -useGui 0 \
   -algorithm "${ALG}" \
   -path $INRIA_PATH/img \
   -train $INRIA_PATH/sigset/train.xml pedModel

br -algorithm pedModel \
   -path $INRIA_PATH/img \
   -enroll $INRIA_PATH/sigset/testSmall.xml pedResults.xml

br -parallelism 0 -algorithm Open+Draw+Show -path $INRIA_PATH/img -enroll pedResults.xml
