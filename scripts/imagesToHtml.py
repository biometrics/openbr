#!/bin/python 
#
# Simple script to output a directory of images into a set of html images. 
# The chrome browser will tile the images well. 

import sys
import glob
import os

assert(len(sys.argv) > 1)

inputDir = sys.argv[1]
if len(sys.argv) > 2:
    imgSize = int(sys.argv[2])
else:
    imgSize = 128

out = open('images.html','w')
imgs = glob.glob(os.path.join(inputDir, '*.jpg'))
imgs.sort()
for i in imgs:
    print >> out, '<img src="%s" height="%d" />' % (i, imgSize)
out.close()


