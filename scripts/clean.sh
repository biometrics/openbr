#!/bin/bash

if [ ! -f clean.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

for File in *.R *.csv *.dot *.duplicate* *.gal *.png *.project *.mask *.mtx *.pdf *.train *.txt Algorithm_Dataset
do
  if [ -e ${File} ]; then
    rm -r ${File}
  fi
done
