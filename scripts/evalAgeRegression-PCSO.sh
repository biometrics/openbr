#!/bin/bash
if [ ! -f evalAgeRegression-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

br -quiet -enroll "../data/PCSO/PCSO.db[query='SELECT File,Age,PersonID FROM PCSO WHERE Age >= 15 AND AGE <= 75', subset=1:200]" terminal.txt > Input.txt

br -algorithm AgeRegression -path ../data/PCSO/Images -enroll Input.txt Output.txt -evalRegression Output.txt Input.txt
