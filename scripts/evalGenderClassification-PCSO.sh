#!/bin/bash
if [ ! -f evalGenderClassification-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

br -quiet -enroll "../data/PCSO/PCSO.db[query='SELECT File,Gender,PersonID FROM PCSO', subset=1:8000]" terminal.txt > Input.txt

br -algorithm GenderClassification -path ../data/PCSO/Images -enroll Input.txt Output.txt -evalClassification Output.txt Input.txt
