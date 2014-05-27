#!/bin/bash

if [ ! -f downloadDatasets.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

# AT&T
if [ ! -d ../data/ATT/img ]; then
  echo "Downloading AT&T…"
  if hash curl 2>/dev/null; then
    curl -OL http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
  else
    wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
  fi

  unzip att_faces.zip -d att_faces
  mv att_faces ../data/ATT/img
  rm ../data/ATT/img/README att_faces.zip
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

# Caltech Pedestrian
if [ ! -d ../data/CaltechPedestrians/vid ]; then
  mkdir ../data/CaltechPedestrians/vid
  echo "Downloading Caltech Pedestrians dataset..."
  prefix="http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA"
  for seq in {0..10}; do
    fname=`printf "set%02d.tar" $seq`
    dlpath="$prefix/$fname"
    if hash curl 2>/dev/null; then
      curl -OL $dlpath
    else
      wget $dlpath
    fi
    tar -xf $fname
  done
  rm *.tar
  ./writeCaltechPedestrianSigset.sh 0 5 train > ../data/CaltechPedestrians/train.xml
  ./writeCaltechPedestrianSigset.sh 6 10 test > ../data/CaltechPedestrians/test.xml
  mv set* ../data/CaltechPedestrians/vid
  if hash curl 2>/dev/null; then
    curl -OL "$prefix/annotations.zip"
  else
    wget "$prefix/annotations.zip"
  fi
  unzip annotations.zip
  rm annotations.zip
  mv annotations ../data/CaltechPedestrians
fi

# INRIA person
if [ ! -d ../data/INRIAPerson/img ]; then
  echo "Downloading INRIA person dataset..."
  if hash curl 2>/dev/null; then
    curl -OL http://pascal.inrialpes.fr/data/human/INRIAPerson.tar
  else
    wget http://pascal.inrialpes.fr/data/human/INRIAPerson.tar
  fi
  tar -xf INRIAPerson.tar
  mkdir ../data/INRIAPerson/img ../data/INRIAPerson/sigset
  ./writeINRIAPersonSigset.sh Train > ../data/INRIAPerson/sigset/train.xml
  ./writeINRIAPersonSigset.sh Test > ../data/INRIAPerson/sigset/test.xml
  ./writeINRIAPersonSigset.sh train_64x128_H96 > ../data/INRIAPerson/sigset/train_normalized.xml
  ./writeINRIAPersonSigset.sh test_64x128_H96 > ../data/INRIAPerson/sigset/test_normalized.xml
  mv INRIAPerson/* ../data/INRIAPerson/img
  rm -r INRIAPerson*
fi

# KTH
if [ ! -d ../data/KTH/vid ]; then
  echo "Downloading KTH..."
  mkdir ../data/KTH/vid
  mkdir ../data/KTH/sigset
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
  rm -f ../data/KTH/vid/boxing/person01_boxing_d4_uncomp.avi
  ./writeKTHSigset.sh 1 16 > ../data/KTH/sigset/train_16ppl.xml
  ./writeKTHSigset.sh 17 25 > ../data/KTH/sigset/test_9ppl.xml
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

./downloadMeds.sh

#LFPW
if [ ! -d ../data/lfpw/trainset ]; then
  echo "Downloading LFPW..."
  if hash curl 2>/dev/null; then
    curl -OL http://ibug.doc.ic.ac.uk/media/uploads/competitions/lfpw.zip
  else
    wget  http://ibug.doc.ic.ac.uk/media/uploads/competitions/lfpw.zip
  fi
  
  unzip lfpw.zip
  mv lfpw ../data
  cd ../data/lfpw
  python ../../scripts/lfpwToSigset.py
  cd ../../scripts
  rm lfpw.zip
fi

