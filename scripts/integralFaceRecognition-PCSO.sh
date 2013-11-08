#!/bin/sh

br -algorithm "{PP5Register+Affine(128,128,0.25,0.35)}+Cvt(Gray)+Gradient+Bin(0,360,9,true)+Merge+Integral+CvtFloat+RecursiveIntegralSampler(4,2,8,LDA(.98)+Normalize(L1))+Cat+PCA(768)+Normalize(L1)+Quantize:UCharL1" -path ~/data/PCSO -train ../data/PCSO/sigset/PCSO_2x1k_train.xml -compare ../data/PCSO/sigset/PCSO_2x1k_test.xml . scores.mtx -eval scores.mtx scores.csv