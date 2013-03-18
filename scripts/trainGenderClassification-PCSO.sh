#!/bin/bash
if [ ! -f trainGenderClassification-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

#rm -f ../share/openbr/models/features/FaceClassificationRegistration
#rm -f ../share/openbr/models/features/FaceClassificationExtraction
rm -f ../share/openbr/models/algorithms/GenderClassification

br -algorithm GenderClassification -path ../data/PCSO/Images -train "../data/PCSO/PCSO.db[query='SELECT File,Gender,PersonID FROM PCSO', subset=0:8000]" ../share/openbr/models/algorithms/GenderClassification
