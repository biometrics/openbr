#!/bin/bash

if [ ! -f helloWorld.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

if ! hash br 2>/dev/null; then
  echo "Can't find 'br'. Did you forget to build and install OpenBR? Here's some help: http://openbiometrics.org/doxygen/latest/installation.html"
  exit
fi

cd ../data

# Download, unzip, and reorganize the BioID database
if [ ! -d BioID/img ]; then
  curl -OL ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip
  unzip BioID-FaceDatabase-V1.2.zip
  mkdir BioID/img
  mv *.pgm BioID/img
  rm *.eye description.txt BioID-FaceDatabase-V1.2.zip
fi

# Download, unzip, and reorganize the MEDS-II database
if [ ! -d MEDS/img ]; then
  curl -OL http://nigos.nist.gov:8080/nist/sd/32/NIST_SD32_MEDS-II_face.zip
  unzip NIST_SD32_MEDS-II_face.zip
  mkdir MEDS/img
  mv data/*/*.jpg MEDS/img
  rm -r data NIST_SD32_MEDS-II_face.zip
fi

# Train the Eigenfaces algorithm
br -algorithm 'Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(64,64,0.25,0.35)+CvtFloat+PCA(0.95):Dist(L2)' -train BioID/img Eigenfaces

# Enroll images using Eigenfaces
br -algorithm Eigenfaces -path MEDS/img -compare MEDS/sigset/MEDS_frontal_target.xml MEDS/sigset/MEDS_frontal_query.xml scores.mtx

# Alternatively, generate galleries first:
# br -algorithm Eigenfaces -path MEDS/img -enroll MEDS/sigset/MEDS_frontal_target.xml target.gal -enroll MEDS/sigset/MEDS_frontal_query.xml query.gal -compare target.gal query.gal scores.mtx

# Evaluate Eigenfaces accuracy
br -eval scores.mtx results.csv -plot results.csv results.pdf
