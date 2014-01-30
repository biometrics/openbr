#!/bin/bash
if [ ! -f trainFaceRecognition-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../share/openbr/models/features/FaceRecognitionRegistration
#rm -f ../share/openbr/models/features/FaceRecognitionExtraction
#rm -f ../share/openbr/models/features/FaceRecognitionEmbedding
#rm -f ../share/openbr/models/features/FaceRecognitionQuantization
#rm -f ../share/openbr/models/algorithms/FaceRecognition

export BR=../build/app/br/br

export PCSO_DIR=/user/pripshare/Databases/FaceDatabases/PCSO/PCSO/

$BR -algorithm FaceRecognition -path "$PCSO_DIR/Images/" -train "$PCSO_DIR/PCSO.db[query='SELECT File,PersonID as Label,PersonID FROM PCSO', subset=0:5:6000]" ../share/openbr/models/algorithms/FaceRecognition

