import torque
import os

def call(alg, name):

    name = 'Attribute_%s' % name
    cmdList = []
    cmdList.append('cd /data2/pattrec/home/bklare/src/openbr/scripts')
    cmdList.append('bash attributesTorque.sh %s %s %s' % (alg,name ,os.environ['DATA']))
    torque.createQsub(3,0,cmdList,name)

call('"CvtFloat+PCA(0.95)+Center(Range)"','Baseline')
call('"CvtFloat+Blur(1.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95)"','DoGLBP')
call('"CvtFloat+Blur(1.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(2,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95)"','DoGLBP2')
call('"CvtFloat+Blur(2.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95)"','DoGLBP_Blur2')


call('"CvtFloat+Blur(1.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(4,4,4,4)+Hist(59)+Cat+PCA(0.95)"','DoGLBP_Rect4')
call('"CvtFloat+Blur(1.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.8)"','DoGLBP_PCA8')
call('"CvtFloat+Blur(1.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95,whiten=true)"','DoGLBP_Whiten')
call('"CvtFloat+Blur(1.1)+GammaFull(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(4,2)+RectRegions(16,16,8,8)+Hist(59)+Cat+PCA(0.95)"','DoGLBP4_Rect16')
