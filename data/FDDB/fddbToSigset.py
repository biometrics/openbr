# Simple script to convert files from the FDDB format into sigsets
# Author: Brendan Klare

import os
import sys
from xml.dom.minidom import Document

if len(sys.argv) < 3:
    print 'Usage: fddbtoSigset.py fddbFileIn xmlSigsetFileOut'
    sys.exit(-1)

inFileName = sys.argv[1]
outFileName = sys.argv[2]
assert(os.path.exists(inFileName))

ini = open(inFileName)
lines = ini.readlines()
ini.close()

xmlDoc = Document()
xmlRoot = xmlDoc.createElement('biometric-signature-set')
i = 0
cnt = 0
while True:
    if not i < len(lines):
        break
    cnt += 1
    line = lines[i]
    xmlImg = xmlDoc.createElement('biometric-signature')
    xmlImg.setAttribute('name','img%05d' % cnt)
    xmlPres = xmlDoc.createElement('presentation')
    xmlPres.setAttribute('file-name',line[:-1])
    xmlPres.setAttribute('Label','pos')

    cnt = int(lines[i+1][:-1])
    for j in range(cnt):
        node = xmlDoc.createElement('data:bbox')
        s = lines[i+2+j][:-1].split()
        radius = float(s[1])
        x = float(s[3])
        y = float(s[4])
        node.setAttribute('x',str(x - radius))
        node.setAttribute('y',str(y - radius))
        node.setAttribute('width',str(radius * 2.0))
        node.setAttribute('height',str(radius * 2.0))
        xmlPres.appendChild(node)
    xmlImg.appendChild(xmlPres)
    xmlRoot.appendChild(xmlImg)
    i += (2 + cnt)

out = open(outFileName,'w')
print >> out, xmlRoot.toprettyxml()
out.close()

