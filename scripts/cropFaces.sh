#!/bin/bash

if [ ! -f classicAlgorithms.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

if ! hash br 2>/dev/null; then
  echo "Can't find 'br'. Did you forget to build and install OpenBR? Here's some help: http://openbiometrics.org/doxygen/latest/installation.html"
  exit
fi

DATA_ROOT=/Volumes/Seagate

for ALGORITHM in CropFace PP5CropFace
do

  #for SIGSET in CUFS_photo CUFS_sketch
  #do
  #  br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/CUFS -enroll ../data/CUFS/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

  #for SIGSET in CUFSF_photo CUFSF_sketch
  #do
  #  br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/CUFSF -enroll ../data/CUFSF/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

  #for SIGSET in fa fb fc dup1 dup2
  #do
  #  br -algorithm ${ALGORITHM} -path "${DATA_ROOT}/FERET/dvd2/gray_feret_cd1/data/images/;${DATA_ROOT}/FERET/dvd2/gray_feret_cd2/data/images/" -enroll ../data/FERET/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

  #for SIGSET in FRGC-1 FRGC-2 FRGC-4_target FRGC-4_query
  #do
  #  br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/FRGC2 -enroll ../data/FRGC/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

  #for SIGSET in HFB_NIR HFB_VIS
  #do
  #  br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/HFB -enroll ../data/HFB/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

  #for SIGSET in LFW
  #do
  #  br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/LFW -enroll ../data/LFW/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

  for SIGSET in MEDS_frontal_all
  do
    br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/MEDS -enroll ../data/MEDS/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  done

  #for SIGSET in PCSO_2x10k
  #do
  #  br -algorithm ${ALGORITHM} -path ${DATA_ROOT}/PCSO -enroll ../data/PCSO/sigset/${SIGSET}.xml galleries/${ALGORITHM}_${SIGSET}.gal
  #done

done