#!/bin/bash

if [ ! -f btas2013.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

mkdir results

# Train and evaluate Fisherfaces
br -algorithm "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+CvtFloat+LDA(0.98)+Normalize(L2):Dist(L2)" -path ~/data/PCSO/ -train ../data/PCSO/sigset/PCSO_2x1k_train.xml Fisherfaces
br -algorithm Fisherfaces -path ~/data/PCSO/ -compare ../data/PCSO/sigset/PCSO_2x1k_test.xml . Fisherfaces.mtx

# Train and evaluate Klarefaces
br -algorithm "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+(Grid(10,10)+SIFTDescriptor(12)+ByRow)/(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))+PCA(0.95)+Normalize(L2)+Cat+Dup(22)+RndSubspace(0.05,1)+LDA(0.98)+Cat+PCA(0.95)+Normalize(L1)+Quantize:NegativeLogPlusOne(ByteL1)" -path ~/data/PCSO/ -train ../data/PCSO/sigset/PCSO_2x1k_train.xml Klarefaces
br -algorithm Klarefaces -path ~/data/PCSO/ -compare ../data/PCSO/sigset/PCSO_2x1k_test.xml . Klarefaces.mtx

# Train and evaluate Klarefaces EBIF
br -algorithm "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+(Grid(10,10)+SIFTDescriptor(12)+ByRow)/(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))/(Resize(64,64)+EBIF)+PCA(0.95)+Normalize(L2)+Cat+Dup(22)+RndSubspace(0.05,1)+LDA(0.98)+Cat+PCA(0.95)+Normalize(L1)+Quantize:NegativeLogPlusOne(ByteL1)" -path ~/data/PCSO/ -train ../data/PCSO/sigset/PCSO_2x1k_train.xml KlarefacesEBIF
br -algorithm KlarefacesEBIF -path ~/data/PCSO/ -compare ../data/PCSO/sigset/PCSO_2x1k_test.xml . KlarefacesEBIF.mtx

# Evaluate and plot
br -makeMask ../data/PCSO/sigset/PCSO_2x1k_test.xml . PCSO.mask
br -eval Fisherfaces.mtx PCSO.mask results/Fisherfaces.csv
br -eval Klarefaces.mtx PCSO.mask results/Klarefaces.csv
br -eval KlarefacesEBIF.mtx PCSO.mask results/KlarefacesEBIF.csv
br -plot results/* plots.pdf