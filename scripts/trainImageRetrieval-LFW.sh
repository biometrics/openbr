#!/bin/bash
if [ ! -f trainImageRetrieval-LFW.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

if [ ! -e Algorithm_Dataset ]; then
  mkdir Algorithm_Dataset
fi

if [ ! -e LFW.mask ]; then
  br -makeMask ~/lfw2[step=10] . LFW.mask
fi

rm -f ../share/openbr/models/algorithms/ImageRetrieval

br -algorithm ImageRetrieval -train ~/lfw2[step=10] ../share/openbr/models/algorithms/ImageRetrieval
br -algorithm ImageRetrieval -compare ~/lfw2[step=10] . ImageRetrieval_LFW.mtx -eval ImageRetrieval_LFW.mtx LFW.mask Algorithm_Dataset/ImageRetrieval_LFW.csv
br -plot Algorithm_Dataset/*_LFW.csv LFW

rm -f ../share/openbr/models/algorithms/ImageRetrieval
