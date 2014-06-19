#!/bin/bash


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
