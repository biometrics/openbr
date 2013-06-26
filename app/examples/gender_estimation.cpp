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
 * \page cli_gender_estimation Gender Estimation
 * \ref cpp_gender_estimation "C++ Equivalent"
 * \code
 * $ br -algorithm GenderEstimation \
 *      -enroll ../data/MEDS/img/S354-01-t10_01.jpg ../data/MEDS/img/S001-01-t10_01.jpg metadata.csv
 * \endcode
 */

//! [gender_estimation]
#include <openbr/openbr_plugin.h>

static void printTemplate(const br::Template &t)
{
    printf("%s gender: %s\n", qPrintable(t.file.fileName()), qPrintable(t.file.get<QString>("Gender")));
}

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);

    // Retrieve class for enrolling templates using the GenderEstimation algorithm
    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("GenderEstimation");

    // Initialize templates
    br::Template queryA("../data/MEDS/img/S354-01-t10_01.jpg");
    br::Template queryB("../data/MEDS/img/S001-01-t10_01.jpg");

    // Enroll templates
    queryA >> *transform;
    queryB >> *transform;

    printTemplate(queryA);
    printTemplate(queryB);

    br::Context::finalize();
    return 0;
}
//! [gender_estimation]
