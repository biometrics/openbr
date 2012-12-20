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

#include <openbr_plugin.h>

#include "frvt2012.h"
#include "core/distance_sse.h"

using namespace br;
using namespace std;

static QSharedPointer<Transform> frvt2012_transform;
static QSharedPointer<Transform> frvt2012_age_transform;
static QSharedPointer<Transform> frvt2012_gender_transform;
static const int frvt2012_template_size = 768;

static void initialize(const string &configuration_location)
{
    if (Globals == NULL) Context::initialize(0, NULL, QString::fromStdString(configuration_location));
    Globals->forceEnrollment = true;
    Globals->quiet = true;
    Globals->parallelism = 0;
}

static Template templateFromONEFACE(const ONEFACE &oneface)
{
    return Template(QString::fromStdString(oneface.description),
                    cv::Mat(oneface.image_height, oneface.image_width, oneface.image_depth == 8 ? CV_8UC1 : CV_8UC3, oneface.data));
}

int32_t get_pid(string &sdk_identifier, string &email_address)
{
    sdk_identifier = "1338";
    email_address = "jklontz@mitre.org";
    return 0;
}

int32_t get_max_template_sizes(uint32_t &max_enrollment_template_size, uint32_t &max_recognition_template_size)
{
    max_enrollment_template_size = frvt2012_template_size;
    max_recognition_template_size = frvt2012_template_size;
    return 0;
}

int32_t initialize_verification(string &configuration_location, vector<string> &descriptions)
{
    (void) descriptions;
    initialize(configuration_location);
    frvt2012_transform = QSharedPointer<Transform>(Transform::make("Cvt(RGBGray)+Cascade(FrontalFace)!<FaceRecognitionRegistration>!<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>", NULL));
    return 0;
}

int32_t convert_multiface_to_enrollment_template(const MULTIFACE &input_faces, uint32_t &template_size, uint8_t *proprietary_template)
{
    uint8_t quality;
    return convert_multiface_to_verification_template(input_faces, template_size, proprietary_template, quality);
}

int32_t convert_multiface_to_verification_template(const MULTIFACE &input_faces, uint32_t &template_size, uint8_t* proprietary_template, uint8_t &quality)
{
    // Enroll templates
    TemplateList templates; templates.reserve(input_faces.size());
    foreach (const ONEFACE &oneface, input_faces)
        templates.append(templateFromONEFACE(oneface));
    templates >> *frvt2012_transform.data();

    // Compute template size
    template_size = templates.size() * frvt2012_template_size;

    // Create proprietary template
    for (int i=0; i<templates.size(); i++)
        memcpy(&proprietary_template[i*frvt2012_template_size], templates[i].m().data, frvt2012_template_size);

    quality = 100;
    return 0;
}

int32_t match_templates(const uint8_t* verification_template, const uint32_t verification_template_size, const uint8_t* enrollment_template, const uint32_t enrollment_template_size, double &similarity)
{
    const int num_verification = verification_template_size / frvt2012_template_size;
    const int num_enrollment = enrollment_template_size / frvt2012_template_size;

    // Return early for failed templates
    if ((num_verification == 0) || (num_enrollment == 0)) {
        similarity = -1;
        return 2;
    }

    similarity = 0;
    for (int i=0; i<num_verification; i++)
        for (int j=0; j<num_enrollment; j++)
            similarity += l1(&verification_template[i*frvt2012_template_size], &enrollment_template[j*frvt2012_template_size], frvt2012_template_size);
    similarity /= num_verification * num_enrollment;
    similarity = std::max(0.0, -0.00112956 * (similarity - 6389.75)); // Yes this is a hard coded hack taken from FaceRecognition score normalization
    return 0;
}

int32_t SdkEstimator::initialize_age_estimation(const string &configuration_location)
{
    initialize(configuration_location);
    frvt2012_age_transform = QSharedPointer<Transform>(Transform::make("Cvt(RGBGray)+Cascade(FrontalFace)!<FaceClassificationRegistration>!<FaceClassificationExtraction>+<AgeRegressor>+Discard", NULL));
    return 0;
}

int32_t SdkEstimator::initialize_gender_estimation(const string &configuration_location)
{
    initialize(configuration_location);
    frvt2012_gender_transform = QSharedPointer<Transform>(Transform::make("Cvt(RGBGray)+Cascade(FrontalFace)!<FaceClassificationRegistration>!<FaceClassificationExtraction>+<GenderClassifier>+Discard", NULL));
    return 0;
}

int32_t SdkEstimator::estimate_age(const ONEFACE &input_face, int32_t &age)
{
    TemplateList templates;
    templates.append(templateFromONEFACE(input_face));
    templates.first().file.setBool("forceEnrollment");
    templates >> *frvt2012_age_transform.data();
    age = templates.first().file.label();
    return templates.first().file.failed() ? 4 : 0;
}

int32_t SdkEstimator::estimate_gender(const ONEFACE &input_face, int8_t &gender, double &mf)
{
    TemplateList templates;
    templates.append(templateFromONEFACE(input_face));
    templates.first().file.setBool("forceEnrollment");
    templates >> *frvt2012_gender_transform.data();
    mf = gender = templates.first().file.label();
    return templates.first().file.failed() ? 4 : 0;
}
