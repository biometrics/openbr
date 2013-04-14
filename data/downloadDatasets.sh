#!/bin/bash

if [ ! -f downloadDatasets.sh ]; then
  echo "Run this script from the data folder!"
  exit
fi

# BioID
if [ ! -d BioID/img ]; then
  echo "Downloading BioID..."
  curl -OL ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip
  unzip BioID-FaceDatabase-V1.2.zip
  mkdir BioID/img
  mv *.pgm BioID/img
  rm *.eye description.txt BioID-FaceDatabase-V1.2.zip
fi

# LFW
if [ ! -d ../data/LFW/img ]; then
  echo "Downloading LFW..."
  curl -OL http://vis-www.cs.umass.edu/lfw/lfw.tgz
  tar -xf lfw.tgz -C LFW
  mv LFW/lfw LFW/img
  rm lfw.tgz
fi

# MEDS
if [ ! -d MEDS/img ]; then
  echo "Downloading MEDS..."
  curl -OL http://nigos.nist.gov:8080/nist/sd/32/NIST_SD32_MEDS-II_face.zip
  unzip NIST_SD32_MEDS-II_face.zip
  mkdir MEDS/img
  mv data/*/*.jpg MEDS/img
  rm -r data NIST_SD32_MEDS-II_face.zip
fi
