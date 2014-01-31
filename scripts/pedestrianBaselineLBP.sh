#!/bin/bash

# Right now this is just a simple proof of concept.

if [ $# -lt 1 ]; then
    echo 'Usage:'
    echo "    $0 show: show the results qualitatively on a small subset."
    echo "    $0 eval: evaluate the results on the full test subset."
    exit 1
fi

# Make sure you set your data path. This will likely by your openbr/data directory.
if [ -z "$DATA" ]; then
    INRIA_PATH=../data/INRIAPerson
else
    INRIA_PATH=$DATA/INRIAPerson
fi

ALG="Open+Cvt(Gray)+BuildScales(Blur(2)+LBP(1,2)+SlidingWindow(Hist(59)+Cat+LDA(isBinary=true),windowWidth=10,takeLargestScale=false,threshold=2),windowWidth=10,takeLargestScale=false,minScale=4)+ConsolidateDetections+Discard"

# Josh's new algorithm (in progress)
# ALG="Open+Cvt(Gray)+Detector(Gradient+Bin(0,360,9,true)+Merge+Integral+IntegralSlidingWindow(RecursiveIntegralSampler(2,2,0,PCA(0.95))+Cat+LDA(0.95,isBinary=true)))"

if [ $1 = 'eval' ]; then 
    TEST=test.xml
else
    TEST=testSmall.xml
fi

br -algorithm "${ALG}" \
   -path $INRIA_PATH/img \
   -train $INRIA_PATH/sigset/train.xml pedModel \
   -enroll $INRIA_PATH/sigset/$TEST pedResults.xml

if [ $1 = 'show' ]; then
    br -parallelism 0 -algorithm Open+Draw+Show -path $INRIA_PATH/img -enroll pedResults.xml
elif [ $1 = 'eval' ]; then
    br -evalDetection pedResults.xml $INRIA_PATH/sigset/$TEST pedEvalResults.csv \
       -plotDetection pedEvalResults.csv pedPlots.pdf
else
    echo "$1 is not a valid command. Choose show or eval."
fi

