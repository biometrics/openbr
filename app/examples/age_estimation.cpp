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
 * \page cli_age_estimation Age Estimation
 * \ref cpp_age_estimation "C++ Equivalent"
 * \code
 * $ br -algorithm AgeEstimation \
 *      -enroll ../data/MEDS/img/S354-01-t10_01.jpg ../data/MEDS/img/S001-01-t10_01.jpg metadata.csv
 * \endcode
 */

//! [age_estimation]
#include <openbr/openbr_plugin.h>

static void printTemplate(const br::Template &t)
{
    printf("%s age: %d\n", qPrintable(t.file.fileName()), int(t.file.get<float>("Age")));
}

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);

    // Retrieve class for enrolling templates using the AgeEstimation algorithm
    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("AgeEstimation");

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
//! [age_estimation]
