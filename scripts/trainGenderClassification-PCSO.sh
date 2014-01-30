#!/bin/bash
if [ ! -f trainGenderClassification-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../share/openbr/models/features/FaceClassificationRegistration
#rm -f ../share/openbr/models/features/FaceClassificationExtraction
#rm -f ../share/openbr/models/algorithms/GenderClassification

export BR=../build/app/br/br
export genderAlg=GenderClassification

export PCSO_DIR=/user/pripshare/Databases/FaceDatabases/PCSO/PCSO/

$BR -algorithm $genderAlg -path $PCSO_DIR/Images -train "$PCSO_DIR/PCSO.db[query='SELECT File,Gender,PersonID FROM PCSO', subset=0:8000]" ../share/openbr/models/algorithms/GenderClassification
