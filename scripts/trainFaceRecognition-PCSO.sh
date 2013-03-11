#!/bin/bash
if [ ! -f trainFaceRecognition-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../models/features/FaceRecognitionRegistration
#rm -f ../models/features/FaceRecognitionExtraction
#rm -f ../models/features/FaceRecognitionEmbedding
#rm -f ../models/features/FaceRecognitionQuantization
rm -f ../models/algorithms/FaceRecognition

br -algorithm FaceRecognition -path ../data/PCSO/Images -train "../data/PCSO/PCSO.db[query='SELECT File,'S'||PersonID,PersonID FROM PCSO', subset=0:5:6000]" ../share/openbr/models/algorithms/FaceRecognition
