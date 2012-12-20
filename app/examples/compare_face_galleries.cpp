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

//! [compare_face_galleries]
// Command line equivalent:
// $ br -algorithm FaceRecognition -forceEnrollment -path ../share/openbr/images/ \
//      -enroll ../share/openbr/images.xml 'images.gal;images.csv[separator=;]' \
//      -compare images.gal images.gal scores.mtx \
//      -convert scores.mtx scores.csv

#include <openbr.h>

int main(int argc, char *argv[])
{
    br_initialize(argc, argv);
    br_set_property("algorithm", "FaceRecognition");
    br_set_property("forceEnrollment", "true");

    // Used to resolve images specified with relative paths
    br_set_property("path", "../share/openbr/images/");

    // 'images.gal' is a binary format used for template comparison, 'images.csv' is a text format used to retrieve template metadata.
    br_enroll("../share/openbr/images.xml", "images.gal;images.csv[separator=;]");

    // Create a binary self-similarity matrix
    br_compare("images.gal", "images.gal", "scores.mtx");

    // Convert the similarity matrix to a readable text format
    br_convert("scores.mtx", "scores.csv");

    br_finalize();
    return 0;
}
//! [compare_face_galleries]
