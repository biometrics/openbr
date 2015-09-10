#!/bin/bash

#input algorithm string as argument
ALGORITHM=$1

if [ ! -f evalFaceRecognition-BLUFR.sh ]; then
	echo "Run this script from the scripts folder!"
	exit
fi

if ! hash br 2>/dev/null; then
	echo "Can't find 'br'. Did you forget to build and install OpenBR? Here's some help: http://openbiometrics.org/docs/install/index.html"
	exit
fi

# Get the data
./downloadDatasets.sh

if [ ! -e Algorithm_Split ]; then
	mkdir Algorithm_Split
fi

for i in `seq 1 10`; do
	br -algorithm ${ALGORITHM} -path ../data/LFW/img -train ../data/LFW/sigset/BLUFR/split${i}/train${i}.xml -compare ../data/LFW/sigset/BLUFR/split${i}/gallery${i}.xml ../data/LFW/sigset/BLUFR/split${i}/probe${i}.xml Algorithm_Split/newAlgorithm_BLUFR${i}.eval
done

br -plot Algorithm_Split/* BLUFR.pdf[smooth=Split]