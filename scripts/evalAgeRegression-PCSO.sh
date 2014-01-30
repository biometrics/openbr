#!/bin/bash
if [ ! -f evalAgeRegression-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

export BR=../build/app/br/br
export PCSO_DIR=/user/pripshare/Databases/FaceDatabases/PCSO/PCSO/
export ageAlg=AgeRegression

# Create a file list by querying the database
$BR -quiet -algorithm Identity -enroll "$PCSO_DIR/PCSO.db[query='SELECT File,Age,PersonID FROM PCSO WHERE Age >= 17 AND AGE <= 68', subset=1:200]" terminal.txt > Input.txt

# Enroll the file list and evaluate performance
$BR -algorithm $ageAlg  -path $PCSO_DIR/Images -enroll Input.txt Output.txt -evalRegression Output.txt Input.txt Age
