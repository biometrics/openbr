#!/bin/bash
if [ ! -f trainAgeRegression-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../share/openbr/models/features/FaceClassificationRegistration
#rm -f ../share/openbr/models/features/FaceClassificationExtraction
rm -f ../share/openbr/models/algorithms/AgeRegression

br -algorithm AgeRegression -path ../data/PCSO/Images -train "../data/PCSO/PCSO.db[query='SELECT File,Age,PersonID FROM PCSO WHERE Age >= 15 AND AGE <= 75', subset=0:200]" ../share/openbr/models/algorithms/AgeRegression
