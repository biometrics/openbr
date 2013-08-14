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

#ifndef FRVT2012_H
#define FRVT2012_H

#include <string>
#include <vector>
#include <stdint.h>
#include <openbr/openbr_export.h>

/*!
 * \defgroup frvt2012 FRVT 2012
 * \brief NIST <a href="http://www.nist.gov/itl/iad/ig/frvt-2012.cfm">Face Recognition Vendor Test 2012</a> API
 */

 /*!
 * \addtogroup frvt2012
 * \{
 */

/*!
 * \brief Data structure representing a single face.
 */
typedef struct sface {
    uint16_t image_width; /*!< \brief Number of pixels horizontally. */
    uint16_t image_height; /*!< \brief Number of pixels vertically. */
    uint16_t image_depth; /*!< \brief Number of bits per pixel. Legal values are 8 and 24. */
    uint8_t format; /*!< \brief Flag indicating native format of the image as supplied to NIST:
                                - 0x01 = JPEG (i.e. compressed data)
                                - 0x02 = PNG (i.e. never compressed data) */
    uint8_t* data; /*!< \brief Pointer to raster scanned data. Either RGB color or intensity.
                               - If image_depth == 24 this points to  3WH bytes  RGBRGBRGB...
                               - If image_depth ==  8 this points to  WH bytes  IIIIIII */
    std::string description; /*!< \brief Single description of the image. */
} ONEFACE;

/*!
 * \brief Data structure representing a set of images from a single person
 *
 * The set of face objects used to pass the image(s) and attribute(s) to
 * the template extraction process.
 */
typedef std::vector<ONEFACE> MULTIFACE;

/*!
 * \brief Return the identifier and contact email address for the software under test.
 *
 * All implementations shall support the self-identification function below.
 * This function is required to support internal NIST book-keeping.
 * The version numbers should be distinct between any versions which offer
 * different algorithmic functionality.
 *
 * \param[out] sdk_identifier
 * Version ID code as hexadecimal integer printed to null terminated ASCII
 * string.  NIST will allocate exactly 5 bytes for this. This will be used to
 * identify the SDK in the results reports.  This value should be changed every
 * time an SDK is submitted to NIST. The value is vendor assigned - format is
 * not regulated by NIST.  EXAMPLE:  "011A"
 *
 * \param[out] email_address
 * Point of contact email address as null terminated ASCII string.
 * NIST will allocate at least 64 bytes for this.  SDK shall not allocate.
 *
 * \return
 *  0 Success
 *  Other Vendor-defined failure
 */
BR_EXPORT int32_t get_pid(std::string &sdk_identifier,
                          std::string &email_address);

/*!
 * \brief Return the maximum template sizes needed during feature extraction.
 *
 * All implementations shall report the maximum expected template sizes.
 * These values will be used by the NIST test harnesses to pre-allocate template
 * data.  The values should apply to a single image. For a MULTIFACE containing
 * K images, NIST will allocate K times the value returned.
 *
 * \param[out] max_enrollment_template_size
 * The maximum possible size, in bytes, of the memory needed to store feature
 * data from a single enrollment image.
 *
 * \param[out] max_recognition_template_size
 * The maximum possible size, in bytes, of the memory needed to store feature
 * data from a single verification or identification image.
 *
 * \return
 *  0 Success
 *  Other Vendor-defined failure
 */
BR_EXPORT int32_t get_max_template_sizes(uint32_t &max_enrollment_template_size,
                                         uint32_t &max_recognition_template_size);

/*!
 * \brief This function initializes the SDK under test.
 *
 * It will be called by the NIST application before any call to the functions
 * convert_multiface_to_enrollment_template or
 * convert_multiface_to_verification_template.
 * The SDK under test should set all parameters. Before any template generation
 * or matching calls are made, the NIST test harness will make a call to the
 * initialization of the function.
 *
 * \param[in] configuration_location
 * A read-only directory containing any vendor-supplied configuration parameters
 * or run-time data files.  The name of this directory is assigned by NIST.
 * It is not hardwired by the provider.  The names of the files in this directory
 * are hardwired in the SDK and are unrestricted.
 *
 * \param[in] descriptions
 * A lexicon of labels one of which will be assigned to each image.
 * EXAMPLE: The descriptions could be {"mugshot", "visa", "unknown"}.
 * These labels are provided to the SDK so that it knows to expect images of
 * these kinds.
 *
 * \return
 *  0 Success
 *  2 Vendor provided configuration files are not readable in the
 * indicated location
 *  8 The descriptions are unexpected or unusable
 *  Other Vendor-defined failure
 */
BR_EXPORT int32_t initialize_verification(std::string &configuration_location,
                                          std::vector<std::string> &descriptions);

/*!
 * \brief This function takes a MULTIFACE, and outputs a proprietary template for enrollment.
 *
 * The memory for the output template is allocated by the NIST test harness
 * before the call i.e. the implementation shall not allocate memory for the result.
 * In all cases, even when unable to extract features, the output shall be a
 * template record that may be passed to the match_templates function without
 * error.  That is, this routine must internally encode "template creation
 * failed" and the matcher must transparently handle this.
 *
 * \param[in] input_faces
 * An instance of a MULTIFACE structure.  Implementations must alter their
 * behavior according to the number of images contained in the structure.
 *
 * \param[in] template_size
 * The size, in bytes, of the output template
 *
 * \param[out] proprietary_template
 * The output template.  The format is entirely unregulated.  NIST will allocate
 * a KT byte buffer for this template: The value K is the number of images in
 * the MULTIFACE; the value T is output by get_max_template_sizes.
 *
 * \return
 *  0 Success
 *  2 Elective refusal to process this kind of MULTIFACE
 *  4 Involuntary failure to extract features (e.g. could not find face in the
 *  input-image)
 *  6 Elective refusal to produce a template (e.g. insufficient pixes between
 *  the eyes)
 *  8 Cannot parse input data (i.e. assertion that input record is
 *  non-conformant)
 *  Other Vendor-defined failure.  Failure codes must be documented and
 * communicated to NIST with the submission of the implementation under test.
 */
BR_EXPORT int32_t convert_multiface_to_enrollment_template(const MULTIFACE &input_faces,
                                                           uint32_t &template_size,
                                                           uint8_t* proprietary_template);

/*!
 * \brief This function takes a MULTIFACE, and outputs a proprietary template for
 * verification.
 *
 * The memory for the output template is allocated by the NIST test harness
 * before the call i.e. the implementation shall not allocate memory for the
 * result.  In all cases, even when unable to extract features, the output shall
 * be a template record that may be passed to the match_templates function
 * without error.  That is, this routine must internally encode "template
 * creation failed" and the matcher must transparently handle this.
 *
 * \param[in] input_faces
 * An instance of a MULTIFACE structure.  Implementations must alter their
 * behavior according to the number of images contained in the structure.
 *
 * \param[in] template_size
 * The size, in bytes, of the output template
 *
 * \param[out] proprietary_template
 * The output template.  The format is entirely unregulated.  NIST will allocate
 * a KT byte buffer for this template: The value K is the number of images in
 * the MULTIFACE; the value T is output by get_max_template_sizes.
 *
 * \param[out] quality
 * An assessment of image quality.  This is optional.  The legal values are
 * - [0,100] - The value should have a monotonic decreasing relationship with
 * false non-match rate anticipated for this sample if it was compared with a
 * pristine image of the same person.  So, a low value indicates high expected
 * FNMR.
 * - 255 - This value indicates a failed attempt to calculate a quality score.
 * - 254 - This values indicates the value was not assigned.
 *
 * \return
 *  0 Success
 *  2 Elective refusal to process this kind of MULTIFACE
 *  4 Involuntary failure to extract features (e.g. could not find face in the
 *  input-image)
 *  6 Elective refusal to produce a template (e.g. insufficient pixes between
 *  the eyes)
 *  8 Cannot parse input data (i.e. assertion that input record is
 *  non-conformant)
 *  Other Vendor-defined failure.  Failure codes must be documented and
 * communicated to NIST with the submission of the implementation under test.
 */
BR_EXPORT int32_t convert_multiface_to_verification_template(const MULTIFACE &input_faces,
                                                             uint32_t &template_size,
                                                             uint8_t* proprietary_template,
                                                             uint8_t &quality);

/*!
 * \brief
 * This function compares two opaque proprietary templates and outputs a
 * similarity score which need not satisfy the metric properties.
 *
 * NIST will allocate memory for this parameter before the call.  When either
 * or both of the input templates are the result of a failed template
 * generation, the similarity score shall be -1 and the function return value
 * shall be 2.
 *
 * \param[in] verification_template
 * A template from convert_multiface_to_verification_template().
 *
 * \param[in] verification_template_size
 * The size, in bytes, of the input verification template 0 <= N <= 2^32 - 1
 *
 * \param[in] enrollment_template
 * A template from convert_multiface_to_enrollment_template().
 *
 * \param[in] enrollment_template_size
 * The size, in bytes, of the input enrollment template  0 <= N <= 2^32 - 1
 *
 * \param[out] similarity
 * A similarity score resulting from comparison of the templates, on the
 * range [0,DBL_MAX].
 *
 * \return
 *  0 Success
 *  2 Either or both of the input templates were result of failed feature
 * extraction
 *  Other Vendor-defined failure.
 */
BR_EXPORT int32_t match_templates(const uint8_t* verification_template,
                                  const uint32_t verification_template_size,
                                  const uint8_t* enrollment_template,
                                  const uint32_t enrollment_template_size,
                                  double &similarity);

/*!
 * \brief Class D estimator abstraction.
 */
struct BR_EXPORT Estimator {
    /*!< */
    static const int NOTIMPLEMENTED = -1;
    virtual ~Estimator() {}

    /*!
     * Intialization functions
     *
     * \param[in] configuration_location
     * 	A read-only directory containing any vendor-supplied configuration
     * 	parameters or run-time data files.
     *
     * \return
     *  0 Success
     *  2 Elective refusal to process this kind of MULTIFACE
     *  4 Involuntary failure to extract features (e.g. could not find face in
     *  the input-image)
     *  8 Cannot parse input data (i.e. assertion that input record is
     *  non-conformant)
     *  Other Vendor-defined failure.  Failure codes must be documented and
     *  communicated to NIST with the submission of the implementation under
     *  test.
     */
    /*!\{*/
    virtual int32_t initialize_frontal_pose_estimation(const std::string &configuration_location) { (void) configuration_location; return NOTIMPLEMENTED; }
    virtual int32_t initialize_age_estimation(const std::string &configuration_location) { (void) configuration_location; return NOTIMPLEMENTED; }
    virtual int32_t initialize_gender_estimation(const std::string &configuration_location) { (void) configuration_location; return NOTIMPLEMENTED; }
    virtual int32_t initialize_expression_estimation(const std::string &configuration_location) { (void) configuration_location; return NOTIMPLEMENTED; }
    /*!\}*/

    /*!
     * Frontal Pose estimation function
     *
     * \param[in] input_face
     * 	An instance of a ONEFACE structure.
     *
     * \param[out] non_frontality
     *  Indication of how far from frontal the head pose is.
     *  The value should be on the range [0,1].
     *
     * \return
     *  0 Success
     *  2 Elective refusal to process this kind of MULTIFACE
     *  4 Involuntary failure to extract features (e.g. could not find face in
     *  the input-image)
     *  8 Cannot parse input data (i.e. assertion that input record is
     *  non-conformant)
     *  Other Vendor-defined failure.  Failure codes must be documented and
     *  communicated to NIST with the submission of the implementation under
     *  test.
     */
    virtual int32_t estimate_frontal_pose_conformance(const ONEFACE &input_face, double &non_frontality) { (void) input_face; (void) non_frontality; return NOTIMPLEMENTED; }

    /*!
     * Age Estimation function
     *
     * \param[in] input_face
     * 	An instance of a ONEFACE structure.
     *
     *  \param[out] age
     *  Indication of the age (in years) of the person.
     *  The value should be on the range [0,100].
     *
     * \return
     *  0 Success
     *  2 Elective refusal to process this kind of MULTIFACE
     *  4 Involuntary failure to extract features (e.g. could not find face in
     *  the input-image)
     *  8 Cannot parse input data (i.e. assertion that input record is
     *  non-conformant)
     *  Other Vendor-defined failure.  Failure codes must be documented and
     *  communicated to NIST with the submission of the implementation under
     *  test.
     */
    virtual int32_t estimate_age(const ONEFACE &input_face, int32_t &age) { (void) input_face; (void) age; return NOTIMPLEMENTED; }

    /*!
     * Gender Estimation function
     *
     * \param[in] input_face
     * 	An instance of a ONEFACE structure.
     *
     * \param[out] gender
     *  Indication of the gender of the person.  Valid values are
     *  0: Male
     *  1: Female
     *  -1: Unknown
     *
     * \param[out] mf
     *  A real-valued measure of maleness-femaleness value on [0,1].
     *  A value of 0 indicates certainty that the subject is a male,
     *  and a value of 1 indicates certainty that the subject is a female.
     *
     * \return
     *  0 Success
     *  2 Elective refusal to process this kind of MULTIFACE
     *  4 Involuntary failure to extract features (e.g. could not find face
     *  in the input-image)
     *  8 Cannot parse input data (i.e. assertion that input record is
     *  non-conformant)
     *  Other Vendor-defined failure.  Failure codes must be documented and
     *  communicated to NIST with the submission of the implementation under test.
     */
    virtual int32_t estimate_gender(const ONEFACE &input_face, int8_t &gender, double &mf) { (void) input_face; (void) gender; (void) mf; return NOTIMPLEMENTED; }

    /*!
     * Expression neutrality function
     *
     * \param[in] input_face
     * 	An instance of a ONEFACE structure.
     *
     * \param[out] expression_neutrality
     *  A real-valued measure of expression neutrality on [0,1] with 0
     *  denoting large deviation from neutral and 1 indicating a fully
     *  neutral expression.
     *
     * \return
     *  0 Success
     *  2 Elective refusal to process this kind of MULTIFACE
     *  4 Involuntary failure to extract features (e.g. could not find face in
     *  the input-image)
     *  8 Cannot parse input data (i.e. assertion that input record is
     *  non-conformant)
     *  Other Vendor-defined failure.  Failure codes must be documented and
     *  communicated to NIST with the submission of the implementation under
     *  test.
     */
    virtual int32_t estimate_expression_neutrality(const ONEFACE &input_face, double &expression_neutrality) { (void) input_face; (void) expression_neutrality; return NOTIMPLEMENTED; }
};

/*!
 * \brief Class D estimator implementation.
 */
struct BR_EXPORT SdkEstimator : public Estimator
{
    /*!
     * \brief Implemented estimators
     */
    /*!\{*/
    int32_t initialize_age_estimation(const std::string &configuration_location);
    int32_t initialize_gender_estimation(const std::string &configuration_location);
    int32_t estimate_age(const ONEFACE &input_face, int32_t &age);
    int32_t estimate_gender(const ONEFACE &input_face, int8_t &gender, double &mf);
    /*!\}*/
};

/*! @}*/

#endif // FRVT2012_H
