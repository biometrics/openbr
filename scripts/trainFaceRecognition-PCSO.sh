#!/bin/bash
if [ ! -f trainFaceRecognition-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../share/openbr/models/features/FaceRecognitionRegistration
#rm -f ../share/openbr/models/features/FaceRecognitionExtraction
#rm -f ../share/openbr/models/features/FaceRecognitionEmbedding
#rm -f ../share/openbr/models/features/FaceRecognitionQuantization
rm -f ../share/openbr/models/algorithms/FaceRecognition

br -algorithm FaceRecognition -path ../data/PCSO/img -train "../data/PCSO/PCSO.db[query='SELECT File,'S'||PersonID,PersonID FROM PCSO', subset=0:5:6000]" ../share/openbr/models/algorithms/FaceRecognition
