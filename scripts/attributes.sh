#!/bin/bash

BASE="Open+PP5Register+Rename(PP5_Landmark0_Right_Eye,Affine_0)+Rename(PP5_Landmark1_Left_Eye,Affine_1)+Affine(192,240,.345,.475)+Cvt(Gray)+Stasm(false,true,[(66.24,114),(125.76,114)])"
SUBSPACE="Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95)"

NOSE="RectFromStasmNoseWithBridge+ROI+Resize(36,24)+$SUBSPACE"
MOUTH="RectFromStasmMouth+ROI+Resize(24,36)+$SUBSPACE"
EYES="RectFromStasmEyes+ROI+Resize(24,36)+$SUBSPACE"
HAIR="RectFromStasmHair+ROI+Resize(24,36)+$SUBSPACE"
BROW="RectFromStasmBrow+ROI+Resize(24,36)+$SUBSPACE"
JAW="RectFromStasmJaw+ROI+Resize(36,36)+$SUBSPACE"
FACE="Crop(24,30,144,190)+Resize(36,36)+$SUBSPACE"

ATTDIR=Attributes
mkdir -p $ATTDIR

# Provide a sensible default value for DATA if undefined
DATA=${DATA:-~/data/CUHK-VHDC}

if [ ! -f $ATTDIR/all.model ]; then
  br -crossValidate 2 -algorithm "CrossValidate($BASE+ \
  ($BROW+ \
    TurkClassifier(eyebrowposition,[closebrows,highbrows],3)/ \
    TurkClassifier(unibrow,[unibrow],3)/ \
    TurkClassifier(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3)/ \
    TurkClassifier(thickeyebrows,[thickeyebrows,lighteyebrows],3))/ \
  ($MOUTH+ \
    TurkClassifier(smiling,[smiling],3)/ \
    TurkClassifier(lipthickness,[cherry,big,small],3)/ \
    TurkClassifier(mouthbite,[underbite,overbite],3)/ \
    TurkClassifier(mouthopen,[closed,noteeth,halfteeth,allteeth],3)/ \
    TurkClassifier(mouthwidth,[small,wide],3)/ \
    TurkClassifier(mustache,[nomustache,linemustache,lightmustache,normalmustache,down],3)/ \
    TurkClassifier(mouthasymmetry,[asymmetrical],3))/ \
  ($EYES+ \
    TurkClassifier(eyeseparation,[close,wide],3)/ \
    TurkClassifier(eyeslant,[slant2,slant1,wild],3)/ \
    TurkClassifier(benteyes,[bent])/ \
    TurkClassifier(eyecolor,[darkeyes,lighteyes],3)/ \
    TurkClassifier(baggyeyes,[baggy],3)/ \
    TurkClassifier(almondeyes,[almond],3)/ \
    TurkClassifier(buriedeyes,[buriedeyes],3)/ \
    TurkClassifier(sleepyeyes,[sleepy],3)/ \
    TurkClassifier(lineeyes,[line],3)/ \
    TurkClassifier(roundeyes,[round],3)/ \
    TurkClassifier(sharpeyes,[sharp],3)/ \
    TurkClassifier(smalleyes,[smalleyes],3)/ \
    TurkClassifier(glasses,[glasses],3)/ \
    TurkClassifier(eyelashvisibility,[feweyelashes],3))/ \
  ($FACE+ \
    TurkClassifier(gender,[male],3)/ \
    TurkClassifier(faceshape,[round,triangular,rectangular],3)/ \
    TurkClassifier(cheekdensity,[puffy,in,normal],3)/ \
    TurkClassifier(facemarks,[scars,moles,normal],3)/ \
    TurkClassifier(facelength,[long],3)/ \
    TurkClassifier(nosetoeyedist,[short,long],3)/ \
    TurkClassifier(nosetomouthdist,[long,small],3))/ \
  ($HAIR+ \
    TurkClassifier(foreheadwrinkles,[wrinkled],3)/ \
    TurkClassifier(foreheadsize,[smallforehead,largeforehead],3)/ \
    TurkClassifier(haircolor,[darkhair,lighthair,greyhair],3)/ \
    TurkClassifier(hairdensity,[thick,bald,thin,halfbald],3)/ \
    TurkClassifier(widowspeak,[widowspeak],3)/ \
    TurkClassifier(hairstyle,[curlyhair],3))/ \
  ($NOSE+ \
    TurkClassifier(noseorientation,[upnose,downnose],3)/ \
    TurkClassifier(nosewidth,[small,thick],3)/ \
    TurkClassifier(nosesize,[smallnose,bignose],3)/ \
    TurkClassifier(brokennose,[broken],3))/ \
  ($JAW+ \
    TurkClassifier(beard,[nobeard,bigbeard,lightbeard,goatee,linebeard,normalbeard,lincolnbeard],3)/ \
    TurkClassifier(chinsize,[shortchin,longchin],3)) \
  ): \
CrossValidate+Fuse([ \
Turk(eyebrowposition,[closebrows,highbrows],3), \
Turk(unibrow,[unibrow],3), \
Turk(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3), \
Turk(thickeyebrows,[thickeyebrows,lighteyebrows],3), \
Turk(smiling,[smiling],3), \
Turk(lipthickness,[cherry,big,small],3), \
Turk(mouthbite,[underbite,overbite],3), \
Turk(mouthopen,[closed,noteeth,halfteeth,allteeth],3), \
Turk(mouthwidth,[small,wide],3), \
Turk(mustache,[nomustache,linemustache,lightmustache,normalmustache,down],3), \
Turk(mouthasymmetry,[asymmetrical],3), \
Turk(eyeseparation,[close,wide],3), \
Turk(eyeslant,[slant2,slant1,wild],3), \
Turk(benteyes,[bent],3), \
Turk(eyecolor,[darkeyes,lighteyes],3), \
Turk(baggyeyes,[baggy],3), \
Turk(almondeyes,[almond],3), \
Turk(buriedeyes,[buriedeyes],3), \
Turk(sleepyeyes,[sleepy],3), \
Turk(lineeyes,[line],3), \
Turk(roundeyes,[round],3), \
Turk(sharpeyes,[sharp],3), \
Turk(smalleyes,[smalleyes],3), \
Turk(glasses,[glasses],3), \
Turk(eyelashvisibility,[feweyelashes],3), \
Turk(gender,[male],3), \
Turk(faceshape,[round,triangular,rectangular],3), \
Turk(cheekdensity,[puffy,in,normal],3), \
Turk(facemarks,[scars,moles,normal],3), \
Turk(facelength,[long],3), \
Turk(nosetoeyedist,[short,long],3), \
Turk(nosetomouthdist,[long,small],3), \
Turk(foreheadwrinkles,[wrinkled],3), \
Turk(foreheadsize,[smallforehead,largeforehead],3), \
Turk(haircolor,[darkhair,lighthair,greyhair],3), \
Turk(hairdensity,[thick,bald,thin,halfbald],3), \
Turk(widowspeak,[widowspeak],3), \
Turk(hairstyle,[curlyhair],3), \
Turk(noseorientation,[upnose,downnose],3), \
Turk(nosewidth,[small,thick],3), \
Turk(nosesize,[smallnose,bignose],3), \
Turk(brokennose,[broken],3), \
Turk(beard,[nobeard,bigbeard,lightbeard,goatee,linebeard,normalbeard,lincolnbeard],3), \
Turk(chinsize,[shortchin,longchin],3)])" \
-path $DATA/CUFSF/target/ -train results1194v2.turk $ATTDIR/all.model
fi

br -crossValidate 2 -path $DATA/CUFSF/target/ -algorithm $ATTDIR/all.model -compare results1194v2.turk results1194v2.turk $ATTDIR/all.mtx
br -crossValidate 2 -setHeader $ATTDIR/all.mtx $DATA/CUFSF/target.xml $DATA/CUFSF/query.xml
br -crossValidate 2 -convert Output $ATTDIR/all.mtx $ATTDIR/all_CUFSF%1.eval
br -plot $ATTDIR/all_CUFSF* results.pdf
