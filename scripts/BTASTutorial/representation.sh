#!/bin/bash

if [ ! -f meds.mask ]; then
  br -makeMask ../../data/MEDS/sigset/MEDS_frontal_target.xml ../../data/MEDS/sigset/MEDS_frontal_query.xml meds.mask
fi

PIXEL_ALG="Open+Cvt(Gray)+Cascade+ASEFEyes+Affine(88,88,0.25,0.35)+Blur(1)+Cat+Normalize(L2):Dist(L2)"
LBP_ALG="Open+Cvt(Gray)+Cascade+ASEFEyes+Affine(88,88,0.25,0.35)+Blur(1)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59)+Normalize(L1)+Cat+Normalize(L2):Dist(L2)"
ALG="Open+Cvt(Gray)+Cascade+ASEFEyes+Affine(88,88,0.25,0.35)+Blur(1)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59)+Normalize(L1)+Cat+LDA(0.95)+Normalize(L2):Dist(L2)"

br -path $DATA/MEDS -algorithm "${PIXEL_ALG}" -compare ../../data/MEDS/sigset/MEDS_frontal_target.xml ../../data/MEDS/sigset/MEDS_frontal_query.xml meds.mtx -eval meds.mtx meds.mask Algorithm_Dataset/pixels_MEDS.csv
br -path $DATA/MEDS -algorithm "${ALG}" -compare ../../data/MEDS/sigset/MEDS_frontal_target.xml ../../data/MEDS/sigset/MEDS_frontal_query.xml meds.mtx -eval meds.mtx meds.mask Algorithm_Dataset/LBP_MEDS.csv

br -path $DATA/LFW-original -algorithm "${ALG}" -train ../../data/LFW/sigset/LFW.xml representLDA.model
br -path $DATA/MEDS -algorithm representLDA.model -compare ../../data/MEDS/sigset/MEDS_frontal_target.xml ../../data/MEDS/sigset/MEDS_frontal_query.xml meds.mtx -eval meds.mtx meds.mask Algorithm_Dataset/LDA_MEDS.csv

br -plot Algorithm_Dataset/* btasResults.pdf
