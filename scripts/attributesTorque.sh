#!/bin/bash


SUBSPACE=$1
NAME=$2
DATA=$3

ROOT=/data2/pattrec/home/bklare/src/openbr/scripts
BR="/data2/pattrec/home/bklare/src/openbr/build/app/br/br"
#SUBSPACE="CvtFloat+PCA(0.95)+Center(Range)"

BASE="Open+PP5Register+Rename(PP5_Landmark0_Right_Eye,Affine_0)+Rename(PP5_Landmark1_Left_Eye,Affine_1)+Affine(192,240,.345,.475)+Cvt(Gray)+Stasm(false,true,[(66.24,114),(125.76,114)])"
NOSE="RectFromStasmNoseWithBridge+ROI+Resize(36,24)+$SUBSPACE"
MOUTH="RectFromStasmMouth+ROI+Resize(24,36)+$SUBSPACE"
EYES="RectFromStasmEyes+ROI+Resize(24,36)+$SUBSPACE"
HAIR="RectFromStasmHair+ROI+Resize(24,36)+$SUBSPACE"
BROW="RectFromStasmBrow+ROI+Resize(24,36)+$SUBSPACE"
JAW="RectFromStasmJaw+ROI+Resize(36,36)+$SUBSPACE"
FACE="Crop(24,30,144,190)+Resize(36,36)+$SUBSPACE"

mkdir $NAME

if [ ! -f $NAME/all.model ]; then
  ${BR} -crossValidate 2 -algorithm "CrossValidate($BASE+ \
($BROW+ \
(Turk(unibrow,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=unibrow,outputVariable=predicted_unibrow)+Cat)/ \
(Turk(eyebroworientation,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=eyebrowsdown,outputVariable=predicted_eyebrowsdown)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=eyebrowsuptodown,outputVariable=predicted_eyebrowsuptodown)+Cat)/ \
(Turk(thickeyebrows,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=thickeyebrows,outputVariable=predicted_thickeyebrows)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=lighteyebrows,outputVariable=predicted_lighteyebrows)+Cat))/ \
($MOUTH+ \
(Turk(smiling,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=smiling,outputVariable=predicted_smiling)+Cat)/ \
(Turk(mouthasymmetry,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=asymmetrical,outputVariable=predicted_asymmetrical)+Cat))/ \
($EYES+ \
(Turk(eyecolor,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=darkeyes,outputVariable=predicted_darkeyes)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=lighteyes,outputVariable=predicted_lighteyes)+Cat)/ \
(Turk(baggyeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=baggy,outputVariable=predicted_baggy)+Cat)/ \
(Turk(almondeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=almond,outputVariable=predicted_almond)+Cat)/ \
(Turk(buriedeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=buriedeyes,outputVariable=predicted_buriedeyes)+Cat)/ \
(Turk(sleepyeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=sleepy,outputVariable=predicted_sleepy)+Cat)/ \
(Turk(lineeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=line,outputVariable=predicted_line)+Cat)/ \
(Turk(roundeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=round,outputVariable=predicted_round)+Cat)/ \
(Turk(smalleyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=smalleyes,outputVariable=predicted_smalleyes)+Cat)/ \
(Turk(glasses,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=glasses,outputVariable=predicted_glasses)+Cat)/ \
(Turk(eyelashvisibility,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=feweyelashes,outputVariable=predicted_feweyelashes)+Cat))/ \
($FACE+ \
(Turk(gender,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=male,outputVariable=predicted_male)+Cat)/ \
(Turk(cheekdensity,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=puffy,outputVariable=predicted_puffy)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=in,outputVariable=predicted_in)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=normal,outputVariable=predicted_normal)+Cat)/ \
(Turk(facemarks,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=scars,outputVariable=predicted_scars)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=moles,outputVariable=predicted_moles)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=normal,outputVariable=predicted_normal)+Cat)/ \
(Turk(facelength,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=long,outputVariable=predicted_long)+Cat)/ \
(Turk(nosetomouthdist,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=long,outputVariable=predicted_long)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=small,outputVariable=predicted_small)+Cat))/ \
($HAIR+ \
(Turk(foreheadwrinkles,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=wrinkled,outputVariable=predicted_wrinkled)+Cat)/ \
(Turk(haircolor,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=darkhair,outputVariable=predicted_darkhair)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=lighthair,outputVariable=predicted_lighthair)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=greyhair,outputVariable=predicted_greyhair)+Cat)/ \
(Turk(hairstyle,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=curlyhair,outputVariable=predicted_curlyhair)+Cat))/ \
($NOSE+ \
(Turk(noseorientation,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=upnose,outputVariable=predicted_upnose)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=downnose,outputVariable=predicted_downnose)+Cat)/ \
(Turk(nosewidth,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=small,outputVariable=predicted_small)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=thick,outputVariable=predicted_thick)+Cat)/ \
(Turk(nosesize,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=smallnose,outputVariable=predicted_smallnose)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=bignose,outputVariable=predicted_bignose)+Cat))/ \
($JAW+ \
(Turk(chinsize,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=shortchin,outputVariable=predicted_shortchin)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=longchin,outputVariable=predicted_longchin))+Cat)): \
CrossValidate+Fuse([ \
Turk(unibrow,[unibrow],3), \
Turk(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3), \
Turk(thickeyebrows,[thickeyebrows,lighteyebrows],3), \
Turk(smiling,[smiling],3), \
Turk(mouthasymmetry,[asymmetrical],3), \
Turk(eyecolor,[darkeyes,lighteyes],3), \
Turk(baggyeyes,[baggy],3), \
Turk(almondeyes,[almond],3), \
Turk(buriedeyes,[buriedeyes],3), \
Turk(sleepyeyes,[sleepy],3), \
Turk(lineeyes,[line],3), \
Turk(roundeyes,[round],3), \
Turk(smalleyes,[smalleyes],3), \
Turk(glasses,[glasses],3), \
Turk(cheekdensity,[puffy,in,normal],3), \
Turk(facemarks,[scars,moles,normal],3), \
Turk(facelength,[long],3), \
Turk(nosetomouthdist,[long,small],3), \
Turk(foreheadwrinkles,[wrinkled],3), \
Turk(haircolor,[darkhair,lighthair,greyhair],3), \
Turk(hairstyle,[curlyhair],3), \
Turk(noseorientation,[upnose,downnose],3), \
Turk(nosewidth,[small,thick],3), \
Turk(nosesize,[smallnose,bignose],3), \
Turk(chinsize,[shortchin,longchin],3)],indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26])" \
-path $DATA/CUFSF/target/ -train results1194v2.turk $NAME/all.model
fi

${BR} -crossValidate 2 -path $DATA/CUFSF/target/ -algorithm $NAME/all.model -compare results1194v2.turk results1194v2.turk $NAME/all.mtx

${BR} -crossValidate 2 -setHeader $NAME/all.mtx $DATA/CUFSF/target.xml $DATA/CUFSF/query.xml
${BR} -crossValidate 2 -convert Output $NAME/all.mtx $NAME/all.rank
${BR} -crossValidate 2 -convert Output $NAME/all.mtx $NAME/all_CUFSF%1.eval
