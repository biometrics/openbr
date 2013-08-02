#!/bin/bash
if [ ! -f evalGenderClassification-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

export BR=../build/app/br/br
export genderAlg=GenderClassification

export PCSO_DIR=/user/pripshare/Databases/FaceDatabases/PCSO/PCSO/

# Create a file list by querying the database
$BR -useGui 0 -quiet -algorithm Identity -enroll "$PCSO_DIR/PCSO.db[query='SELECT File,Gender,PersonID FROM PCSO', subset=1:8000]" terminal.txt > Input.txt

# Enroll the file list and evaluate performance
$BR -useGui 0  -algorithm $genderAlg -path $PCSO_DIR/Images -enroll Input.txt Output.txt -evalClassification Output.txt Input.txt Gender