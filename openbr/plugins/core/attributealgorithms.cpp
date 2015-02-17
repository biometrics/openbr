/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initializes global abbreviations with implemented algorithms for attributes
 * \author Babatunde Ogunfemi \cite baba1472
 */
class AttributeAlgorithmsInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        // Constants
        QString BASE="Open+PP5Register+Rename(PP5_Landmark0_Right_Eye,Affine_0)+Rename(PP5_Landmark1_Left_Eye,Affine_1)+Affine(192,240,.345,.475)+Cvt(Gray)+Stasm(false,true,[(66.24,114),(125.76,114)])";
        QString SUBSPACE ="Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95)";

        QString NOSE="RectFromStasmNoseWithBridge+ROI+Resize(36,24)+" + SUBSPACE;
        QString MOUTH="RectFromStasmMouth+ROI+Resize(24,36)+" + SUBSPACE;
        QString EYES="RectFromStasmEyes+ROI+Resize(24,36)+" + SUBSPACE;
        QString HAIR="RectFromStasmHair+ROI+Resize(24,36)+" + SUBSPACE;
        QString BROW="RectFromStasmBrow+ROI+Resize(24,36)+" + SUBSPACE;
        QString JAW="RectFromStasmJaw+ROI+Resize(36,36)+" + SUBSPACE;
        QString FACE = "Crop(24,30,144,190)+Resize(36,36)+" +  SUBSPACE;

        // All Attributes
        Globals->abbreviations.insert("AllAttributes", "AttributeBrow/AttributeMouth/AttributeEyes/AttributeFace/AttributeHair/AttributeNose/AttributeJaw");
        Globals->abbreviations.insert("AllAttributesMatching", "(AttributeBrow)/(AttributeMouth)/(AttributeEyes)/(AttributeFace)/(AttributeHair)/(AttributeNose)/(AttributeJaw):AttributeMatch");

        //Individual Attributes
        Globals->abbreviations.insert("AttributeBrow", "(" + BASE+ "+" + BROW + "+"
        "TurkClassifier(eyebrowposition,[closebrows,highbrows],3)/"
        "TurkClassifier(unibrow,[unibrow],3)/"
        "TurkClassifier(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3)/"
        "TurkClassifier(thickeyebrows,[thickeyebrows,lighteyebrows],3))");
        Globals->abbreviations.insert("AttributeMouth", "(" + BASE + "+" + MOUTH + "+"
        "TurkClassifier(smiling,[smiling],3)/"
        "TurkClassifier(lipthickness,[cherry,big,small],3)/"
        "TurkClassifier(mouthbite,[underbite,overbite],3)/"
        "TurkClassifier(mouthopen,[closed,noteeth,halfteeth,allteeth],3)/"
        "TurkClassifier(mouthwidth,[small,wide],3)/"
        "TurkClassifier(mustache,[nomustache,linemustache,lightmustache,normalmustache,down],3)/"
        "TurkClassifier(mouthasymmetry,[asymmetrical],3))");
        Globals->abbreviations.insert("AttributeEyes", "(" + BASE + "+" + EYES + "+ "
        "TurkClassifier(eyeseparation,[close,wide],3)/"
        "TurkClassifier(eyeslant,[slant2,slant1,wild],3)/"
        "TurkClassifier(benteyes,[bent])/"
        "TurkClassifier(eyecolor,[darkeyes,lighteyes],3)/"
        "TurkClassifier(baggyeyes,[baggy],3)/"
        "TurkClassifier(almondeyes,[almond],3)/"
        "TurkClassifier(buriedeyes,[buriedeyes],3)/"
        "TurkClassifier(sleepyeyes,[sleepy],3)/"
        "TurkClassifier(lineeyes,[line],3)/"
        "TurkClassifier(roundeyes,[round],3)/"
        "TurkClassifier(sharpeyes,[sharp],3)/"
        "TurkClassifier(smalleyes,[smalleyes],3)/"
        "TurkClassifier(glasses,[glasses],3)/"
        "TurkClassifier(eyelashvisibility,[feweyelashes],3))");
        Globals->abbreviations.insert("AttributeFace", "(" + BASE + "+" + FACE + "+"
        "TurkClassifier(gender,[male],3)/"
        "TurkClassifier(faceshape,[round,triangular,rectangular],3)/"
        "TurkClassifier(cheekdensity,[puffy,in,normal],3)/"
        "TurkClassifier(facemarks,[scars,moles,normal],3)/"
        "TurkClassifier(facelength,[long],3)/"
        "TurkClassifier(nosetoeyedist,[short,long],3)/"
        "TurkClassifier(nosetomouthdist,[long,small],3))");
        Globals->abbreviations.insert("AttributeHair", "(" + BASE + "+" + HAIR + "+"
        "TurkClassifier(foreheadwrinkles,[wrinkled],3)/"
        "TurkClassifier(foreheadsize,[smallforehead,largeforehead],3)/"
        "TurkClassifier(haircolor,[darkhair,lighthair,greyhair],3)/"
        "TurkClassifier(hairdensity,[thick,bald,thin,halfbald],3)/"
        "TurkClassifier(widowspeak,[widowspeak],3)/"
        "TurkClassifier(hairstyle,[curlyhair],3))");
        Globals->abbreviations.insert("AttributeNose", "(" + BASE + "+" + NOSE + "+"
        "TurkClassifier(noseorientation,[upnose,downnose],3)/"
        "TurkClassifier(nosewidth,[small,thick],3)/"
        "TurkClassifier(nosesize,[smallnose,bignose],3)/"
        "TurkClassifier(brokennose,[broken],3))");
        Globals->abbreviations.insert("AttributeJaw", "(" + BASE + "+" + JAW + "+"
        "TurkClassifier(beard,[nobeard,bigbeard,lightbeard,goatee,linebeard,normalbeard,lincolnbeard],3)/"
        "TurkClassifier(chinsize,[shortchin,longchin],3))");
        Globals->abbreviations.insert("AttributeMatch", "Fuse(["
        "Turk(eyebrowposition,[closebrows,highbrows],3),"
        "Turk(unibrow,[unibrow],3),"
        "Turk(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3),"
        "Turk(thickeyebrows,[thickeyebrows,lighteyebrows],3),"
        "Turk(smiling,[smiling],3),"
        "Turk(lipthickness,[cherry,big,small],3),"
        "Turk(mouthbite,[underbite,overbite],3),"
        "Turk(mouthopen,[closed,noteeth,halfteeth,allteeth],3),"
        "Turk(mouthwidth,[small,wide],3),"
        "Turk(mustache,[nomustache,linemustache,lightmustache,normalmustache,down],3),"
        "Turk(mouthasymmetry,[asymmetrical],3),"
        "Turk(eyeseparation,[close,wide],3),"
        "Turk(eyeslant,[slant2,slant1,wild],3),"
        "Turk(benteyes,[bent],3),"
        "Turk(eyecolor,[darkeyes,lighteyes],3),"
        "Turk(baggyeyes,[baggy],3),"
        "Turk(almondeyes,[almond],3),"
        "Turk(buriedeyes,[buriedeyes],3),"
        "Turk(sleepyeyes,[sleepy],3),"
        "Turk(lineeyes,[line],3),"
        "Turk(roundeyes,[round],3),"
        "Turk(sharpeyes,[sharp],3),"
        "Turk(smalleyes,[smalleyes],3),"
        "Turk(glasses,[glasses],3),"
        "Turk(eyelashvisibility,[feweyelashes],3),"
        "Turk(gender,[male],3),"
        "Turk(faceshape,[round,triangular,rectangular],3),"
        "Turk(cheekdensity,[puffy,in,normal],3),"
        "Turk(facemarks,[scars,moles,normal],3),"
        "Turk(facelength,[long],3),"
        "Turk(nosetoeyedist,[short,long],3),"
        "Turk(nosetomouthdist,[long,small],3),"
        "Turk(foreheadwrinkles,[wrinkled],3),"
        "Turk(foreheadsize,[smallforehead,largeforehead],3),"
        "Turk(haircolor,[darkhair,lighthair,greyhair],3),"
        "Turk(hairdensity,[thick,bald,thin,halfbald],3),"
        "Turk(widowspeak,[widowspeak],3),"
        "Turk(hairstyle,[curlyhair],3),"
        "Turk(noseorientation,[upnose,downnose],3),"
        "Turk(nosewidth,[small,thick],3),"
        "Turk(nosesize,[smallnose,bignose],3),"
        "Turk(brokennose,[broken],3),"
        "Turk(beard,[nobeard,bigbeard,lightbeard,goatee,linebeard,normalbeard,lincolnbeard],3),"
        "Turk(chinsize,[shortchin,longchin],3)])");
    }
};

BR_REGISTER(Initializer, AttributeAlgorithmsInitializer)

} // namespace br

#include "core/attributealgorithms.moc"
