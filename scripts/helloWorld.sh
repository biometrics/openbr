#!/bin/bash

if [ ! -f helloWorld.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

if ! hash br 2>/dev/null; then
  echo "Can't find 'br'. Did you forget to build and install OpenBR? Here's some help: http://openbiometrics.org/doxygen/latest/installation.html"
  exit
fi

# Get the data
./downloadDatasets.sh

# Train the Eigenfaces algorithm
br -algorithm 'Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+CvtFloat+PCA(0.95):Dist(L2)' -train BioID/img Eigenfaces

# Enroll images using Eigenfaces
br -algorithm Eigenfaces -path MEDS/img -compare MEDS/sigset/MEDS_frontal_target.xml MEDS/sigset/MEDS_frontal_query.xml scores.mtx

# Alternatively, generate galleries first:
# br -algorithm Eigenfaces -path MEDS/img -enroll MEDS/sigset/MEDS_frontal_target.xml target.gal -enroll MEDS/sigset/MEDS_frontal_query.xml query.gal -compare target.gal query.gal scores.mtx

# Evaluate Eigenfaces accuracy
br -eval scores.mtx results.csv -plot results.csv results.pdf

echo "Not very accurate right? That's why nobody uses Eigenfaces anymore!"
