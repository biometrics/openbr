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
 * \brief Initializes global abbreviations with implemented algorithms
 * \author Josh Klontz \cite jklontz
 */
class AlgorithmsInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        // Face
        Globals->abbreviations.insert("FaceRecognition", "FaceDetection+FaceRecognitionRegistration+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>+SetMetadata(AlgorithmID,-1):Unit(ByteL1)");
        Globals->abbreviations.insert("GenderClassification", "FaceDetection+Expand+FaceClassificationRegistration+Expand+<FaceClassificationExtraction>+<GenderClassifier>+Discard");
        Globals->abbreviations.insert("AgeRegression", "FaceDetection+Expand+FaceClassificationRegistration+Expand+<FaceClassificationExtraction>+<AgeRegressor>+Discard");
        Globals->abbreviations.insert("MedianFace", "Open+Expand+Cascade(FrontalFace)+ASEFEyes+Affine(256,256,0.37,0.45)+Center(Median)");
        Globals->abbreviations.insert("BlurredFaceDetection", "Open+LimitSize(1024)+SkinMask/(Cvt(Gray)+GradientMask)+And+Morph(Erode,16)+LargestConvexArea");
        Globals->abbreviations.insert("DrawFaceDetection", "Open+Cascade(FrontalFace)+Expand+ASEFEyes+Draw(inPlace=true)");
        Globals->abbreviations.insert("ShowFaceDetection", "DrawFaceDetection+Contract+First+Show+Discard");
        Globals->abbreviations.insert("DownloadFaceRecognition", "Download+Open+ROI+Cvt(Gray)+Cascade(FrontalFace)+FaceRecognitionRegistration+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>+SetMetadata(AlgorithmID,-1):Unit(ByteL1)");
        Globals->abbreviations.insert("OpenBR", "FaceRecognition");
        Globals->abbreviations.insert("GenderEstimation", "GenderClassification");
        Globals->abbreviations.insert("AgeEstimation", "AgeRegression");
        Globals->abbreviations.insert("FaceRecognition2", "{PP5Register+Affine(128,128,0.25,0.35)+Cvt(Gray)}+(Gradient+HistBin(0,360,9,true))/(Blur(1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2,true)+HistBin(0,10,10,true))+Merge+Integral+RecursiveIntegralSampler(4,2,8,LDA(.98)+Normalize(L1))+Cat+PCA(768)+Normalize(L1)+Quantize:UCharL1");
        Globals->abbreviations.insert("CropFace", "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.25,0.35)");
        Globals->abbreviations.insert("4SF", "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+(Grid(10,10)+SIFTDescriptor(12)+ByRow)/(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))+PCA(0.95)+Cat+Normalize(L2)+Dup(12)+RndSubspace(0.05,1)+LDA(0.98)+Cat+PCA(0.95)+Normalize(L1)+Quantize:NegativeLogPlusOne(ByteL1)");

        // Video
        Globals->abbreviations.insert("DisplayVideo", "FPSLimit(30)+Show(false,[FrameNumber])+Discard");
        Globals->abbreviations.insert("PerFrameDetection", "SaveMat(original)+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+RestoreMat(original)+Draw(inPlace=true)+Show(false,[FrameNumber])+Discard");
        Globals->abbreviations.insert("AgeGenderDemo", "SaveMat(original)+Cvt(Gray)+Cascade(FrontalFace)+Expand+FaceClassificationRegistration+<FaceClassificationExtraction>+<AgeRegressor>/<GenderClassifier>+Discard+RestoreMat(original)+Draw(inPlace=true)+DrawPropertiesPoint([Age,Gender],Affine_0,inPlace=true)+SaveMat(original)+Discard+Contract+RestoreMat(original)+FPSCalc+Show(false,[AvgFPS,Age,Gender])+Discard");
        Globals->abbreviations.insert("ShowOpticalFlowField", "SaveMat(original)+AggregateFrames(2)+OpticalFlow(useMagnitude=false)+Grid(100,100)+DrawOpticalFlow+FPSLimit(30)+Show(false)+Discard");
        Globals->abbreviations.insert("ShowOpticalFlowMagnitude", "AggregateFrames(2)+OpticalFlow+Normalize(Range,false,0,255)+Cvt(Color)+Draw+FPSLimit(30)+Show(false)+Discard");
        Globals->abbreviations.insert("ShowMotionSegmentation", "DropFrames(5)+AggregateFrames(2)+OpticalFlow+CvtUChar+WatershedSegmentation+DrawSegmentation+Draw+FPSLimit(30)+Show(false)+Discard");

        Globals->abbreviations.insert("HOGVideo", "Stream(DropFrames(5)+Cvt(Gray)+Grid(5,5)+ROIFromPts(32,24)+Expand+Resize(32,32)+Gradient+RectRegions+HistBin(0,360,8)+Hist(8)+Cat)+Contract+CatRows+KMeans(500)+Hist(500)+SVM");
        Globals->abbreviations.insert("HOFVideo", "Stream(DropFrames(5)+Grid(5,5)+AggregateFrames(2)+OpticalFlow+ROIFromPts(32,24)+Expand+Resize(32,32)+Gradient+RectRegions+HistBin(0,360,8)+Hist(8)+Cat)+Contract+CatRows+KMeans(500)+Hist(500)");
        Globals->abbreviations.insert("HOGHOFVideo", "Stream(DropFrames(5)+Grid(5,5)+AggregateFrames(2)+(OpticalFlow+ROIFromPts(32,24)+Expand+Resize(32,32)+Gradient+RectRegions+HistBin(0,360,8)+Hist(8)+Cat+Contract)/(First+Cvt(Gray)+ROIFromPts(32,24)+Expand+Resize(32,32)+Gradient+RectRegions+HistBin(0,360,8)+Hist(8)+Cat+Contract)+CatCols)+Contract+CatRows+KMeans(500)+Hist(500)+SVM");

        // Generic Image Processing
        Globals->abbreviations.insert("SIFT", "Open+KeyPointDetector(SIFT)+KeyPointDescriptor(SIFT):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("SURF", "Open+KeyPointDetector(SURF)+KeyPointDescriptor(SURF):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("SmallSIFT", "Open+LimitSize(512)+KeyPointDetector(SIFT)+KeyPointDescriptor(SIFT):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("SmallSURF", "Open+LimitSize(512)+KeyPointDetector(SURF)+KeyPointDescriptor(SURF):KeyPointMatcher(BruteForce)");
        Globals->abbreviations.insert("ColorHist", "Open+LimitSize(512)+Expand+EnsureChannels(3)+SplitChannels+Hist(256,0,8)+Cat+Normalize(L1):L2");
        Globals->abbreviations.insert("ImageSimilarity", "Open+EnsureChannels(3)+Resize(256,256)+SplitChannels+RectRegions(64,64,64,64)+Hist(256,0,8)+Cat:NegativeLogPlusOne(L2)");
        Globals->abbreviations.insert("ImageClassification", "Open+CropSquare+LimitSize(256)+Cvt(Gray)+Gradient+HistBin(0,360,9,true)+Merge+Integral+RecursiveIntegralSampler(4,2,8,Singleton(KMeans(256)))+Cat+CvtFloat+Hist(256)+KNN(5,Dist(L1),false,5)+Rename(KNN,Subject)");
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
        Globals->abbreviations.insert("FlipiBug","Flip+ReorderPoints([ 0, 1, 2, 3, 4, 5, 6, 7, 17,18,19,20,21,31,32,36,37,38,39,40,41,48,49,50,55,56,60,61,65], \
                                                                     [16,15,14,13,12,11,10, 9, 26,25,24,23,22,35,34,45,44,43,42,47,46,54,53,52,59,58,64,63,67])");
        Globals->abbreviations.insert("FlipiBugNoJaw","Flip+ReorderPoints([0,1,2,3,4,14,15,19,20,21,22,23,24,31,32,33,38,39,43,44,48], \
                                                                          [9,8,7,6,5,18,17,28,27,26,25,30,29,37,36,35,42,41,47,46,50])");
        Globals->abbreviations.insert("FlipMUCT","Flip+ReorderPoints([ 0, 1, 2, 3, 4,5,6,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,46,48,49,50,65,60,59,58,68,69,70,71], \
                                                                     [14,13,12,11,10,9,8,15,16,17,18,19,20,32,33,34,35,36,45,44,43,42,47,54,53,52,63,62,55,56,72,73,74,75])");

        // Transforms
        Globals->abbreviations.insert("FaceDetection", "Open+Cvt(Gray)+Cascade(FrontalFace)");
        Globals->abbreviations.insert("DenseLBP", "(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))");
        Globals->abbreviations.insert("DenseHOG", "Gradient+RectRegions(8,8,6,6)+HistBin(0,360,8)+Hist(8)");
        Globals->abbreviations.insert("DenseSIFT", "(Grid(10,10)+SIFTDescriptor(12)+ByRow)");
        Globals->abbreviations.insert("DenseSIFT2", "(Grid(5,5)+SIFTDescriptor(12)+ByRow)");
        Globals->abbreviations.insert("FaceRecognitionRegistration", "ASEFEyes+Affine(88,88,0.25,0.35)");
        Globals->abbreviations.insert("FaceRecognitionExtraction", "(Mask+DenseSIFT/DenseLBP+DownsampleTraining(PCA(0.95),instances=1)+Normalize(L2)+Cat)");
        Globals->abbreviations.insert("FaceRecognitionEmbedding", "(Dup(12)+RndSubspace(0.05,1)+DownsampleTraining(LDA(0.98),instances=-2)+Cat+DownsampleTraining(PCA(768),instances=1))");
        Globals->abbreviations.insert("FaceRecognitionQuantization", "(Normalize(L1)+Quantize)");
        Globals->abbreviations.insert("FaceClassificationRegistration", "ASEFEyes+Affine(56,72,0.33,0.45)");
        Globals->abbreviations.insert("FaceClassificationExtraction", "((Grid(7,7)+SIFTDescriptor(8)+ByRow)/DenseLBP+DownsampleTraining(PCA(0.95),instances=-1, inputVariable=Gender)+Cat)");
        Globals->abbreviations.insert("AgeRegressor", "DownsampleTraining(Center(Range),instances=-1, inputVariable=Age)+DownsampleTraining(SVM(RBF,EPS_SVR,inputVariable=Age),instances=100, inputVariable=Age)");
        Globals->abbreviations.insert("GenderClassifier", "DownsampleTraining(Center(Range),instances=-1, inputVariable=Gender)+DownsampleTraining(SVM(RBF,C_SVC,inputVariable=Gender),instances=4000, inputVariable=Gender)");
        Globals->abbreviations.insert("UCharL1", "Unit(ByteL1)");
    }
};

BR_REGISTER(Initializer, AlgorithmsInitializer)

} // namespace br

#include "core/algorithms.moc"
