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
 * \page cli_face_recognition_evaluation Face Recognition Evaluation
 * \ref c_face_recognition_evaluation "C Equivalent"
 * \code
 * $ br -algorithm FaceRecognition -path ../data/MEDS/img/ \
 *      -enroll ../data/MEDS/sigset/MEDS_frontal_target.xml target.gal \
 *      -enroll ../data/MEDS/sigset/MEDS_frontal_query.xml query.gal \
 *      -compare target.gal query.gal scores.mtx \
 *      -makeMask ../data/MEDS/sigset/MEDS_frontal_target.xml ../data/MEDS/sigset/MEDS_frontal_query.xml MEDS.mask \
 *      -eval scores.mtx MEDS.mask Algorithm_Dataset/FaceRecognition_MEDS.csv \
 *      -plot Algorithm_Dataset/FaceRecognition_MEDS.csv MEDS
 * \endcode
 */

//! [face_recognition_evaluation]
#include <openbr/openbr.h>

int main(int argc, char *argv[])
{
    br_initialize(argc, argv);

    // Equivalent to 'Globals->algorithm = "FaceRecognition";' in C++ API
    br_set_property("algorithm", "FaceRecognition");

    // Equivalent to 'Globals->path = "../data/MEDS/img/";' in C++ API
    br_set_property("path", "../data/MEDS/img/");

    // Enroll galleries, don't re-enroll if they already exist (cache)
    br_enroll("../data/MEDS/sigset/MEDS_frontal_target.xml", "target.gal[cache]");
    br_enroll("../data/MEDS/sigset/MEDS_frontal_query.xml", "query.gal[cache]");

    // Compare galleries and store result in a binary similarity matrix
    br_compare("target.gal", "query.gal", "FaceRecognition_MEDS.mtx");

    // Make a ground truth "mask" matrix from the sigsets.
    br_make_mask("../data/MEDS/sigset/MEDS_frontal_target.xml", "../data/MEDS/sigset/MEDS_frontal_query.xml", "MEDS.mask");

    // Evaluate the performance of OpenBR's FaceRecognition and a COTS face recognition system.
    br_eval("FaceRecognition_MEDS.mtx", "MEDS.mask", "Algorithm_Dataset/FaceRecognition_MEDS.csv");

    // The '_' character has special significance and is used to populate plot legends.
    // Requires R installation, see documentation of br_plot for details.
    const char *files[1];
    files[0] = "Algorithm_Dataset/FaceRecognition_MEDS.csv";
    br_plot(1, files, "MEDS", true);

    br_finalize();
    return 0;
}
//! [face_recognition_evaluation]
