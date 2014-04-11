#!/bin/bash

BASE="Open+GroundTruth(../../sigsets/CUHK-VHDC/CUFSF/target.xml,[NEC3RightEye,NEC3LeftEye])+Rename(NEC3RightEye,Affine_0)+Rename(NEC3LeftEye,Affine_1)+Affine(192,240,.345,.475)+Cvt(Gray)"
SUBSPACE="Normalize(L2)+PCA(0.95)+Center(Range)"
NOSE="RectFromStasmNoseWithBridge+ROI+Resize(76,52)+$SUBSPACE"
MOUTH="RectFromStasmMouth+ROI+Resize(36,104)+$SUBSPACE"
EYES="RectFromStasmEyes+ROI+Resize(24,136)+$SUBSPACE"
HAIR="RectFromStasmHair+ROI+Resize(60,116)+$SUBSPACE"
BROW="RectFromStasmBrow+ROI+Resize(24,136)+$SUBSPACE"
JAW="RectFromStasmJaw+ROI+Resize(104,164)+$SUBSPACE"
FACE="Cascade(FrontalFace)+Resize(104,104)+$SUBSPACE"

mkdir -p models
rm models/all

if [ ! -f models/all ]; then
  br -crossValidate 2 -algorithm "CrossValidate($BASE+Stasm(false,true,[(66.24,114),(125.76,114)])+ \
($BROW+ \
(Turk(unibrow,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=hasunibrow,outputVariable=predicted_hasunibrow)+Cat)/ \
(Turk(eyebroworientation,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=eyebrowsdown,outputVariable=predicted_eyebrowsdown)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=eyebrowsuptodown,outputVariable=predicted_eyebrowsuptodown)+Cat)/ \
(Turk(eyebrowthickness,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=thickeyebrows,outputVariable=predicted_thickeyebrows)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=lighteyebrows,outputVariable=predicted_lighteyebrows)+Cat))/ \
($MOUTH+ \
(Turk(expression,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=smiling,outputVariable=predicted_smiling)+Cat)/ \
(Turk(mouthasymmetry,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=asymmetrical,outputVariable=predicted_asymmetrical)+Cat))/ \
($EYES+ \
(Turk(eyecolor,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=darkeyes,outputVariable=predicted_darkeyes)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=lighteyes,outputVariable=predicted_lighteyes)+Cat)/ \
(Turk(baggyeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=baggy,outputVariable=predicted_baggy)+Cat)/ \
(Turk(almondeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=almond,outputVariable=predicted_almond)+Cat)/ \
(Turk(buriedeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=buried,outputVariable=predicted_buried)+Cat)/ \
(Turk(sleepyeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=sleepy,outputVariable=predicted_sleepy)+Cat)/ \
(Turk(lineeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=line,outputVariable=predicted_line)+Cat)/ \
(Turk(roundeyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=round,outputVariable=predicted_round)+Cat)/ \
(Turk(smalleyes,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=hassmalleyes,outputVariable=predicted_hassmalleyes)+Cat)/ \
(Turk(glasses,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=hasglasses,outputVariable=predicted_hasglasses)+Cat)/ \
(Turk(eyelashvisibility,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=feweyelashes,outputVariable=predicted_feweyelashes)+Cat))/ \
($FACE+ \
(Turk(gender,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=male,outputVariable=predicted_male)+Cat)/ \
(Turk(cheekdensity,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=puffy,outputVariable=predicted_puffy)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=in,outputVariable=predicted_in)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=normal,outputVariable=predicted_normal)+Cat)/ \
(Turk(facemarks,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=scars,outputVariable=predicted_scars)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=moles,outputVariable=predicted_moles)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=normal,outputVariable=predicted_normal)+Cat)/ \
(Turk(facelength,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=longface,outputVariable=predicted_longface)+Cat)/ \
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
(Turk(nosewidth,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=narrow,outputVariable=predicted_narrow)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=thick,outputVariable=predicted_thick)+Cat)/ \
(Turk(nosesize,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=smallnose,outputVariable=predicted_smallnose)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=bignose,outputVariable=predicted_bignose)+Cat))/ \
($JAW+ \
(Turk(chinsize,3)+SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=shortchin,outputVariable=predicted_shortchin)/ \
SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=longchin,outputVariable=predicted_longchin))+Cat)): \
CrossValidate+Fuse([ \
Turk(unibrow,[hasunibrow],3), \
Turk(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3), \
Turk(eyebrowthickness,[thickeyebrows,lighteyebrows],3), \
Turk(expression,[smiling],3), \
Turk(mouthasymmetry,[asymmetrical],3), \
Turk(eyecolor,[darkeyes,lighteyes],3), \
Turk(baggyeyes,[baggy],3), \
Turk(almondeyes,[almond],3), \
Turk(buriedeyes,[buried],3), \
Turk(sleepyeyes,[sleepy],3), \
Turk(lineeyes,[line],3), \
Turk(roundeyes,[round],3), \
Turk(smalleyes,[hassmalleyes],3), \
Turk(glasses,[hasglasses],3), \
Turk(cheekdensity,[puffy,in,normal],3), \
Turk(facemarks,[scars,moles,normal],3), \
Turk(facelength,[longface],3), \
Turk(nosetomouthdist,[long,small],3), \
Turk(foreheadwrinkles,[wrinkled],3), \
Turk(haircolor,[darkhair,lighthair,greyhair],3), \
Turk(hairstyle,[curlyhair],3), \
Turk(noseorientation,[upnose,downnose],3), \
Turk(nosewidth,[narrow,thick],3), \
Turk(nosesize,[smallnose,bignose],3), \
Turk(chinsize,[shortchin,longchin],3)],indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26])" \
-path ../../img/CUHK-VHDC/CUFSF/target/ -train results30v2.turk models/all 
fi

br -crossValidate 2 -path ../../img/CUHK-VHDC/CUFSF/target/ -algorithm models/all -compare results30v2.turk results30v2.turk simmat/all.mtx

br -crossValidate 2 -setHeader simmat/all.mtx ../../sigsets/CUHK-VHDC/CUFSF/target.xml ../../sigsets/CUHK-VHDC/CUFSF/query.xml
br -crossValidate 2 -convert Output simmat/all.mtx output/all.rank
br -crossValidate 2 -convert Output simmat/all.mtx algorithm_dataset/all_CUFSF%1.eval

# Not trained on: earpitch, earsize, neck thickness
# Not used for comparison: gender, eyelashvisbility

#br -crossValidate 2 -path ../../img/CUHK-VHDC/CUFSF/target/ -algorithm models/attributes -enroll results30v2.turk gallery/results30v2.gal

#for attribute in male puffy in scars moles longface long small darkeyes lighteyes upnose downnose darkhair lighthair thickeyebrows lighteyebrows upnose downnose narrow thick smallnose bignose asymmetrical eyebrowsdown eyebrowsuptodown shortchin longchin baggy almond buried sleepy line round hassmalleyes smiling feweyelashes wrinkled curlyhair hasunibrow; do

#echo $attribute
#br -evalRegression gallery/results30v2.gal gallery/results30v2.gal $attribute predicted_$attribute

#done
