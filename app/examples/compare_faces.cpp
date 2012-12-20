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

//! [compare_faces]
// Command line equivalent:
// $ br -algorithm FaceRecognition -forceEnrollment \
//      -compare ../share/openbr/images/S354-01-t10_01.jpg ../share/openbr/images/S354-02-t10_01.jpg \
//      -compare ../share/openbr/images/S024-01-t10_01.jpg ../share/openbr/images/S354-02-t10_01.jpg

#include <openbr.h>

int main(int argc, char *argv[])
{
    br_initialize(argc, argv);

    // Specify how to enroll and compare images
    br_set_property("algorithm", "FaceRecognition");

    // Enroll exactly one template per image
    br_set_property("forceEnrollment", "true");

    // Images taken from MEDS-II dataset: http://www.nist.gov/itl/iad/ig/sd32.cfm
    br_compare("../share/openbr/images/S354-01-t10_01.jpg", "../share/openbr/images/S354-02-t10_01.jpg");
    br_compare("../share/openbr/images/S024-01-t10_01.jpg", "../share/openbr/images/S354-02-t10_01.jpg");

    br_finalize();
    return 0;
}
//! [compare_faces]
