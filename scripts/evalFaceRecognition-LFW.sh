#!/bin/bash

ALGORITHM=FaceRecognition

if [ ! -f evalFaceRecognition-LFW.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

if ! hash br 2>/dev/null; then
  echo "Can't find 'br'. Did you forget to build and install OpenBR? Here's some help: http://openbiometrics.org/doxygen/latest/installation.html"
  exit
fi

# Get the data
./downloadDatasets.sh

if [ ! -e Algorithm_Dataset ]; then
  mkdir Algorithm_Dataset
fi

# Run the LFW test protocol
br -algorithm $ALGORITHM -path ../data/LFW/img/ -crossValidate 10 -pairwiseCompare ../data/LFW/sigset/test_image_restricted_target.xml ../data/LFW/sigset/test_image_restricted_query.xml ${ALGORITHM}_LFW.mtx -convert Output ${ALGORITHM}_LFW.mtx Algorithm_Dataset/${ALGORITHM}_LFW%1.eval

# Plot results
br -plot Algorithm_Dataset/* 'lfw_results.pdf[smooth=Dataset,rocOptions=[yLimits=(0,1)]]'
