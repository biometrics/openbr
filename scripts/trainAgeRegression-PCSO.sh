#!/bin/bash
if [ ! -f trainAgeRegression-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../share/openbr/models/features/FaceClassificationRegistration
#rm -f ../share/openbr/models/features/FaceClassificationExtraction
#rm -f ../share/openbr/models/algorithms/AgeRegression

export BR=../build/app/br/br
export ageAlg=AgeRegression

export PCSO_DIR=/user/pripshare/Databases/FaceDatabases/PCSO/PCSO/

$BR -algorithm $ageAlg -path $PCSO_DIR/Images -train "$PCSO_DIR/PCSO.db[query='SELECT File,Age,PersonID FROM PCSO WHERE Age >= 17 AND AGE <= 68', subset=0:200]" ../share/openbr/models/algorithms/AgeRegression
