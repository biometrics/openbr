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

//! [evaluate_face_recognition]
// Command line equivalent:
// $ mkdir Algorithm_Dataset
// $ br -makeMask ../share/openbr/images.xml ../share/openbr/images.xml images.mask \
//      -eval images.mtx images.mask Algorithm_Dataset/FaceRecognition_MEDS.csv results \
//      -plot AlgorithmDataset

#include <openbr.h>

int main(int argc, char *argv[])
{
    br_initialize(argc, argv);

    // Make a self-similar ground truth "mask" matrix from a sigset.
    br_make_mask("../share/openbr/images.xml", "../share/openbr/images.xml", "scores.mask");

    // First run "compare_face_galleries" to generate "scores.mtx"
    br_eval("scores.mtx", "scores.mask", "Algorithm_Dataset/FaceRecognition_MEDS.csv");

    // Requires R installation, see documentation of br_plot for details
    const char *files[1];
    files[0] = "Algorithm_Dataset/FaceRecognition_MEDS.csv";
    br_plot(1, files, "results", true);

    br_finalize();
    return 0;
}
//! [evaluate_face_recognition]
