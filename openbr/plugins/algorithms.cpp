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

#include "openbr_internal.h"

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initializes global abbreviations with implemented algorithms
 * \author Josh Klontz \cite jklontz
 */
class AlgorithmsInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        // Face
        Globals->abbreviations.insert("FaceRecognition", "FaceDetection!<FaceRecognitionRegistration>!<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>:MatchProbability(ByteL1)");
        Globals->abbreviations.insert("GenderClassification", "FaceDetection!<FaceClassificationRegistration>!<FaceClassificationExtraction>+<GenderClassifier>+Discard");
        Globals->abbreviations.insert("AgeRegression", "FaceDetection!<FaceClassificationRegistration>!<FaceClassificationExtraction>+<AgeRegressor>+Discard");
        Globals->abbreviations.insert("FaceQuality", "Open!Cascade(FrontalFace)+ASEFEyes+Affine(64,64,0.25,0.35)+ImageQuality+Cvt(Gray)+DFFS+Discard");
        Globals->abbreviations.insert("MedianFace", "Open!Cascade(FrontalFace)+ASEFEyes+Affine(256,256,0.37,0.45)+Center(Median)");
        Globals->abbreviations.insert("BlurredFaceDetection", "Open+LimitSize(1024)+SkinMask/(Cvt(Gray)+GradientMask)+And+Morph(Erode,16)+LargestConvexArea");
        Globals->abbreviations.insert("DrawFaceDetection", "Open+Cascade(FrontalFace)!ASEFEyes+Draw");
        Globals->abbreviations.insert("ShowFaceDetection", "DrawFaceDetection!Show[distribute=false]");
        Globals->abbreviations.insert("OpenBR", "FaceRecognition");
        Globals->abbreviations.insert("GenderEstimation", "GenderClassification");
        Globals->abbreviations.insert("AgeEstimation", "AgeRegression");
        Globals->abbreviations.insert("FaceRecognition2", "{PP5Register+Affine(128,128,0.25,0.35)+Cvt(Gray)}+(Gradient+Bin(0,360,9,true))/(Blur(1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2,true)+Bin(0,10,10,true))+Merge+Integral+RecursiveIntegralSampler(4,2,8,LDA(.98)+Normalize(L1))+Cat+PCA(768)+Normalize(L1)+Quantize:UCharL1");
        Globals->abbreviations.insert("CropFace", "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.25,0.35)");

        // Video
        Globals->abbreviations.insert("DisplayVideo", "Stream(FPSLimit(30)+Show(false,[FrameNumber])+Discard)");
        Globals->abbreviations.insert("PerFrameDetection", "Stream(SaveMat(original)+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+RestoreMat(original)+Draw(inPlace=true)+Show(false,[FrameNumber])+Discard)");
        Globals->abbreviations.insert("AgeGenderDemo", "Stream(SaveMat(original)+Cvt(Gray)+Cascade(FrontalFace)+Expand+<FaceClassificationRegistration>+<FaceClassificationExtraction>+<AgeRegressor>/<GenderClassifier>+Discard+RestoreMat(original)+Draw(inPlace=true)+DrawPropertiesPoint([Age,Gender],Affine_0,inPlace=true)+SaveMat(original)+Discard+Contract+RestoreMat(original)+FPSCalc+Show(false,[AvgFPS,Age,Gender])+Discard)");

        Globals->abbreviations.insert("HOG", "Stream(DropFrames(5)+Cvt(Gray)+KeyPointDetector(SIFT)+ROI+Expand+Resize(32,32)+Gradient+RectRegions+Bin(0,360,8)+Hist(8)+Cat)+Contract+CatRows+KMeans(500)+Hist(500)+SVM");
        Globals->abbreviations.insert("HOF", "Stream(DropFrames(5)+KeyPointDetector(SIFT)+AggregateFrames(2)+OpticalFlow+ROI+Expand+Resize(32,32)+Gradient+RectRegions+Bin(0,360,8)+Hist(8)+Cat)+Contract+CatRows+KMeans(500)+Hist(500)");
        Globals->abbreviations.insert("HOGHOF", "Stream(DropFrames(5)+KeyPointDetector(SIFT)+AggregateFrames(2)+(OpticalFlow++ROI+Expand+Resize(32,32)+Gradient+RectRegions+Bin(0,360,8)+Hist(8)+Cat+Contract)/(First+Cvt(Gray)+ROI+Expand+Resize(32,32)+Gradient+RectRegions+Bin(0,360,8)+Hist(8)+Cat+Contract)+CatCols)+Contract+CatRows+KMeans(500)+Hist(500)+SVM");

        // Generic Image Processing
        Globals->abbreviations.insert("SIFT", "Open+KeyPointDetector(SIFT)+KeyPointDescriptor(SIFT):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("SURF", "Open+KeyPointDetector(SURF)+KeyPointDescriptor(SURF):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("SmallSIFT", "Open+LimitSize(512)+KeyPointDetector(SIFT)+KeyPointDescriptor(SIFT):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("SmallSURF", "Open+LimitSize(512)+KeyPointDetector(SURF)+KeyPointDescriptor(SURF):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("ColorHist", "Open+LimitSize(512)!EnsureChannels(3)+SplitChannels+Hist(256,0,8)+Cat+Normalize(L1):L2");
        Globals->abbreviations.insert("ImageClassification", "Open+CropSquare+LimitSize(256)+Cvt(Gray)+Gradient+Bin(0,360,9,true)+Merge+Integral+RecursiveIntegralSampler(4,2,8,Singleton(KMeans(256)))+Cat+CvtFloat+Hist(256)+KNN(5,Dist(L1),false,5)+Rename(KNN,Subject)");
        Globals->abbreviations.insert("TanTriggs", "Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)");

        // Hash
        Globals->abbreviations.insert("FileName", "Name+Identity:Identical");
        Globals->abbreviations.insert("MD5", "Open+CryptographicHash(Md5):Identical");
        Globals->abbreviations.insert("SHA1", "Open+CryptographicHash(Sha1):Identical");

        // Miscellaneous
        Globals->abbreviations.insert("Display", "Open+Identity+Show+Discard");
        Globals->abbreviations.insert("RegisterAffine", "Open+Affine(256,256,0.37,0.45)");
        Globals->abbreviations.insert("ContrastEnhanced", "Open+Affine(256,256,0.37,0.45)+Cvt(Gray)+Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)");
        Globals->abbreviations.insert("ColoredLBP", "Open+Affine(128,128,0.37,0.45)+Cvt(Gray)+Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+ColoredU2");

        // Transforms
        Globals->abbreviations.insert("FaceDetection", "(Open+Cvt(Gray)+Cascade(FrontalFace))");
        Globals->abbreviations.insert("DenseLBP", "(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))");
        Globals->abbreviations.insert("DenseSIFT", "(Grid(10,10)+SIFTDescriptor(12)+ByRow)");
        Globals->abbreviations.insert("FaceRecognitionRegistration", "(ASEFEyes+Affine(88,88,0.25,0.35)+DownsampleTraining(FTE(DFFS),instances=1))");
        Globals->abbreviations.insert("FaceRecognitionExtraction", "(Mask+DenseSIFT/DenseLBP+DownsampleTraining(PCA(0.95),instances=1)+Normalize(L2)+Cat)");
        Globals->abbreviations.insert("FaceRecognitionEmbedding", "(Dup(12)+RndSubspace(0.05,1)+DownsampleTraining(LDA(0.98),instances=-2)+Cat+DownsampleTraining(PCA(768),instances=1))");
        Globals->abbreviations.insert("FaceRecognitionQuantization", "(Normalize(L1)+Quantize)");
        Globals->abbreviations.insert("FaceClassificationRegistration", "(ASEFEyes+Affine(56,72,0.33,0.45)+FTE(DFFS))");
        Globals->abbreviations.insert("FaceClassificationExtraction", "((Grid(7,7)+SIFTDescriptor(8)+ByRow)/DenseLBP+DownsampleTraining(PCA(0.95),instances=-1, inputVariable=Gender)+Cat)");
        Globals->abbreviations.insert("AgeRegressor", "DownsampleTraining(Center(Range),instances=-1, inputVariable=Age)+DownsampleTraining(SVM(RBF,EPS_SVR,inputVariable=Age),instances=100, inputVariable=Age)");
        Globals->abbreviations.insert("GenderClassifier", "DownsampleTraining(Center(Range),instances=-1, inputVariable=Gender)+DownsampleTraining(SVM(RBF,C_SVC,inputVariable=Gender),instances=4000, inputVariable=Gender)");
        Globals->abbreviations.insert("UCharL1", "Unit(ByteL1)");
    }
};

BR_REGISTER(Initializer, AlgorithmsInitializer)

} // namespace br

#include "algorithms.moc"
