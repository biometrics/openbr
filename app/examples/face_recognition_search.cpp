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
 * \page cli_face_recognition_search Face Recognition Search
 * \ref cpp_face_recognition_search "C++ Equivalent"
 * \code
 * $ br -algorithm FaceRecognition -enrollAll -enroll ../data/MEDS/img 'meds.gal;meds.csv[separator=;]'
 * $ br -algorithm FaceRecognition -compare meds.gal ../data/MEDS/img/S001-01-t10_01.jpg match_scores.csv
 * \endcode
 */

//! [face_recognition_search]
#include <openbr/openbr_plugin.h>

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);

    // Retrieve classes for enrolling and comparing templates using the FaceRecognition algorithm
    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("FaceRecognition");
    QSharedPointer<br::Distance> distance = br::Distance::fromAlgorithm("FaceRecognition");

    // Initialize templates
    br::TemplateList target = br::TemplateList::fromGallery("../data/MEDS/img");
    br::Template query("../data/MEDS/img/S001-01-t10_01.jpg");

    // Enroll templates
    br::Globals->enrollAll = true; // Enroll 0 or more faces per image
    target >> *transform;
    br::Globals->enrollAll = false; // Enroll exactly one face per image
    query >> *transform;

    // Compare templates
    QList<float> scores = distance->compare(target, query);

    // Print an example score
    printf("Images %s and %s have a match score of %.3f\n",
           qPrintable(target[3].file.name),
           qPrintable(query.file.name),
           scores[3]);

    br::Context::finalize();
    return 0;
}
//! [face_recognition_search]
