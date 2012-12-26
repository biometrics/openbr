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
 * \page cli_compare_faces Compare Faces
 * \ref cpp_compare_faces "Source Equivalent"
 * \code
 * $ br -algorithm FaceRecognition \
 *      -compare ../share/openbr/images/S354-01-t10_01.jpg ../share/openbr/images/S354-02-t10_01.jpg \
 *      -compare ../share/openbr/images/S024-01-t10_01.jpg ../share/openbr/images/S354-02-t10_01.jpg
 * \endcode
 */

//! [compare_faces]
#include <openbr_plugin.h>

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);

    // Retrieve classes for enrolling and comparing templates using the FaceRecognition algorithm
    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("FaceRecognition");
    QSharedPointer<br::Distance> distance = br::Distance::fromAlgorithm("FaceRecognition");

    // Initialize templates
    br::Template queryA("../data/MEDS/img/S354-02-t10_01.jpg");
    br::Template queryB("../data/MEDS/img/S386-04-t10_01.jpg");
    br::Template target("../data/MEDS/img/S354-01-t10_01.jpg");

    // Enroll templates
    queryA >> *transform;
    queryB >> *transform;
    target >> *transform;

    // Compare templates
    float comparisonA = distance->compare(target, queryA);
    float comparisonB = distance->compare(target, queryB);

    printf("Genuine match score: %.3f\n", comparisonA); // Scores above 1 are strong matches
    printf("Impostor match score: %.3f\n", comparisonB); // Scores below 0.5 are strong non-matches

    br::Context::finalize();
    return 0;
}
//! [compare_faces]
