#!/bin/bash
if [ ! -f evalGenderClassification-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

# Create a file list by querying the database
br -quiet -algorithm Identity -enroll "../data/PCSO/PCSO.db[query='SELECT File,Gender,PersonID FROM PCSO', subset=1:8000]" terminal.txt > Input.txt

# Enroll the file list and evaluate performance
br -algorithm GenderClassification -path ../data/PCSO/img -enroll Input.txt Output.txt -evalClassification Output.txt Input.txt
