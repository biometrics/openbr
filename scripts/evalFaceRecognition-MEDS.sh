#!/bin/bash

ALGORITHM=FaceRecognition

if [ ! -f evalFaceRecognition-MEDS.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

if ! hash br 2>/dev/null; then
  echo "Can't find 'br'. Did you forget to build and install OpenBR? Here's some help: http://openbiometrics.org/doxygen/latest/installation.html"
  exit
fi

# Get the data
./downloadMEDS.sh

if [ ! -e Algorithm_Dataset ]; then
  mkdir Algorithm_Dataset
fi

if [ ! -e MEDS.mask ]; then
  br -makeMask ../data/MEDS/sigset/MEDS_frontal_target.xml ../data/MEDS/sigset/MEDS_frontal_query.xml MEDS.mask
fi

# Run Algorithm on MEDS
br -algorithm ${ALGORITHM} -path ../data/MEDS/img -compare ../data/MEDS/sigset/MEDS_frontal_target.xml ../data/MEDS/sigset/MEDS_frontal_query.xml ${ALGORITHM}_MEDS.mtx -eval ${ALGORITHM}_MEDS.mtx MEDS.mask Algorithm_Dataset/${ALGORITHM}_MEDS.csv

# Plot results
br -plot Algorithm_Dataset/*_MEDS.csv MEDS
