#!/bin/bash

if [ ! -f downloadDatasets.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

# BioID
if [ ! -d ../data/BioID/img ]; then
  echo "Downloading BioID..."
  curl -OL ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip
  unzip BioID-FaceDatabase-V1.2.zip
  mkdir ../data/BioID/img
  mv *.pgm ../data/BioID/img
  rm *.eye description.txt BioID-FaceDatabase-V1.2.zip
fi

# LFW
if [ ! -d ../data/LFW/img ]; then
  echo "Downloading LFW..."
  curl -OL http://vis-www.cs.umass.edu/lfw/lfw.tgz
  tar -xf lfw.tgz
  mv lfw ../data/LFW/img
  rm lfw.tgz
fi

# MEDS
if [ ! -d ../data/MEDS/img ]; then
  echo "Downloading MEDS..."
  curl -OL http://nigos.nist.gov:8080/nist/sd/32/NIST_SD32_MEDS-II_face.zip
  unzip NIST_SD32_MEDS-II_face.zip
  mkdir ../data/MEDS/img
  mv data/*/*.jpg ../data/MEDS/img
  rm -r data NIST_SD32_MEDS-II_face.zip
fi
