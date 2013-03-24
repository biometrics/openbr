#!/bin/bash
if [ ! -f trainFaceRecognition-MEDS.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

rm -f ../share/openbr/models/algorithms/FaceRecognitionHoG

br -algorithm FaceRecognitionHoG -path ../data/MEDS/img -train ../data/MEDS/sigset/MEDS_frontal_all.xml ../share/openbr/models/algorithms/FaceRecognitionHoG
