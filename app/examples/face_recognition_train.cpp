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

/*!
 * \ingroup cli
 * \page cli_face_recognition_train Face Recognition Train
 * \ref cpp_face_recognition_train "C++ Equivalent"
 * \code
 * $ br -algorithm 'Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+(Grid(10,10)+SIFTDescriptor(12)+ByRow)/(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))+PCA(0.95)+Normalize(L2)+Dup(12)+RndSubspace(0.05,1)+LDA(0.98)+Cat+PCA(0.95)+Normalize(L1)+Quantize:NegativeLogPlusOne(ByteL1)' -train ../data/ATT/img FaceRecognitionATT
 * \endcode
 */

//! [face_recognition_train]
#include <openbr/openbr_plugin.h>

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);

    const QString trainedModelFile = "FaceRecognitionATT";
    if (QFile(trainedModelFile).exists())
        return 0; // Already trained

    br::Globals->algorithm = "Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+(Grid(10,10)+SIFTDescriptor(12)+ByRow)/(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))+PCA(0.95)+Normalize(L2)+Dup(12)+RndSubspace(0.05,1)+LDA(0.98)+Cat+PCA(0.95)+Normalize(L1)+Quantize:NegativeLogPlusOne(ByteL1)";
    // br::Globals->algorithm = "4SF"; // Equally valid alternative. "4SF" is the abbreviation, see openbr/plugins/algorithms.cpp

    // Note the structure of the `../data/ATT/img` training data:
    //  - Subdirectory for each person
    //  - Multiple images per person
    // Run `scripts/downloadDatasets.sh` to obtain these images if you haven't already.
    const QString trainingData = "../data/ATT/img";

    // After training completes you can use `FaceRecognitionATT` like `FaceRecognition`:
    //  $ br -algorithm FaceRecognitionATT
    // provided the `FaceRecognitionATT` file is in your current working directory or in `share/openbr/models/algorithms`.
    printf("Note: Training will take at least a few minutes to complete.\n");
    br::Train(trainingData, trainedModelFile);

    br::Context::finalize();
    return 0;
}
//! [face_recognition_train]
