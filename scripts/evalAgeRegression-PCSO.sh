#!/bin/bash
if [ ! -f evalAgeRegression-PCSO.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

# Create a file list by querying the database
br -quiet -algorithm Identity -enroll "../data/PCSO/PCSO.db[query='SELECT File,Age,PersonID FROM PCSO WHERE Age >= 15 AND AGE <= 75', subset=1:200]" terminal.txt > Input.txt

# Enroll the file list and evaluate performance
br -algorithm AgeRegression -path ../data/PCSO/img -enroll Input.txt Output.txt -evalRegression Output.txt Input.txt
