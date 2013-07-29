#!/bin/bash

if [ ! -f downloadDatasets.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

# BioID
if [ ! -d ../data/BioID/img ]; then
  echo "Downloading BioID..."
  if hash curl 2>/dev/null; then
    curl -OL ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip
  else
    wget ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip
  fi

  unzip BioID-FaceDatabase-V1.2.zip
  mkdir ../data/BioID/img
  mv *.pgm ../data/BioID/img
  rm *.eye description.txt BioID-FaceDatabase-V1.2.zip
fi

# LFW
if [ ! -d ../data/LFW/img ]; then
  echo "Downloading LFW..."
  if hash curl 2>/dev/null; then
    curl -OL http://vis-www.cs.umass.edu/lfw/lfw.tgz
  else
    wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
  fi

  tar -xf lfw.tgz
  mv lfw ../data/LFW/img
  rm lfw.tgz
fi

# MEDS
if [ ! -d ../data/MEDS/img ]; then
  echo "Downloading MEDS..."
  if hash curl 2>/dev/null; then
    curl -OL http://nigos.nist.gov:8080/nist/sd/32/NIST_SD32_MEDS-II_face.zip
  else
    wget http://nigos.nist.gov:8080/nist/sd/32/NIST_SD32_MEDS-II_face.zip
  fi

  unzip NIST_SD32_MEDS-II_face.zip
  mkdir ../data/MEDS/img
  mv data/*/*.jpg ../data/MEDS/img
  rm -r data NIST_SD32_MEDS-II_face.zip
fi

# KTH
if [ ! -d ../data/KTH/vid ]; then
  echo "Downloading KTH..."
  mkdir ../data/KTH/vid
  for vidclass in {'boxing','handclapping','handwaving','jogging','running','walking'}; do
    if hash curl 2>/dev/null; then
      curl -OL http://www.nada.kth.se/cvap/actions/${vidclass}.zip
    else
      wget http://www.nada.kth.se/cvap/actions/${vidclass}.zip
    fi
    mkdir ../data/KTH/vid/${vidclass}
    unzip ${vidclass}.zip -d ../data/KTH/vid/${vidclass}
	rm ${vidclass}.zip
  done
  # this file is corrupted
  rm ../data/KTH/vid/boxing/person01_boxing_d4_uncomp.avi
fi
