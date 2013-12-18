#!/usr/bin/python

# This scripts converts the LFPW .pts files into xml sigsets that can be readily 
#   used within openbr. 

from xml.dom.minidom import Document
import glob
import os

for lfpwType in ['train','test']:
    files = glob.glob('%sset/*.pts' % lfpwType)
    files.sort()
    cnt = 0
    
    xmlDoc = Document()
    xmlRoot = xmlDoc.createElement('biometric-signature-set')
    for ptsFile in files:
        cnt += 1
        ini = open(ptsFile)
        lines = ini.readlines()
        ini.close()

        pntStrList = []
        n = int(lines[1].split()[1])
        for i in range(3,3+n):
            pntStrList.append('(%s)' % (','.join(lines[i].split())))
        pntStr = '[%s]' % ','.join(pntStrList)

        xmlSubj = xmlDoc.createElement('biometric-signature')
        xmlSubj.setAttribute('name','subj_%05d' % cnt) 
        xmlPres = xmlDoc.createElement('presentation')
        xmlPres.setAttribute('file-name','%sset/%s.png' % (lfpwType,os.path.splitext(ptsFile)[0]))
        xmlPres.setAttribute('Points',pntStr)
        xmlSubj.appendChild(xmlPres)
        xmlRoot.appendChild(xmlSubj)
    
    if not os.path.exists('sigset'):
        os.mkdir('sigset')
    out = open('sigset/%s.xml' % lfpwType,'w')
    out.write(xmlRoot.toprettyxml())
    out.close()


