#!/bin/bash

if [ ! -f clean.sh ]; then
  echo "Run this script from the scripts folder!"
  exit
fi

rm -rf *.R *.csv *.dot *.duplicate* *.gal *.png *.project *.mask *.mtx *.pdf *.train *.txt Algorithm*
