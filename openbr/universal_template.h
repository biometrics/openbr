/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2014 Noblis                                                     *
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

#ifndef BR_UNIVERSAL_TEMPLATE_H
#define BR_UNIVERSAL_TEMPLATE_H

#include <stdio.h>
#include <stdint.h>
#include <openbr/openbr_export.h>

#ifdef __cplusplus
extern "C" {
#endif

// Disable 'nonstandard extension used : zero-sized array in struct/union' warning
#ifdef _MSC_VER
#  pragma warning(disable: 4200)
#endif // _MSC_VER

/*!
 * \brief A flat template format for representing arbitrary feature vectors.
 */
struct br_universal_template
{
    int32_t algorithmID; /*!< Interpretation of _data_ after _mdSize_. */
    uint32_t frame; /*!< Video frame number, or <tt>numeric_limits<uint32_t>::max()</tt> for still images. */
    int32_t x; /*!< Region of interest horizontal offset (pixels). */
    int32_t y; /*!< Region of interest vertical offset (pixels). */
    uint32_t width;  /*!< Region of interest horizontal size (pixels). */
    uint32_t height; /*!< Region of interest vertical size (pixels). */
    float confidence; /*!< Region of interest confidence. */
    uint32_t personID; /*!< Unique identifier or <tt>numeric_limits<uint32_t>::max()</tt> if unknown. */
    uint32_t mdSize; /*!< Length of a null-terminated metadata string at the beginning of _data_,
                          including the null-terminator character itself. */
    uint32_t fvSize; /*!< Length of the feature vector after the metadata in _data_. */
    unsigned char data[]; /*!< (_mdSize_ + _fvSize_)-byte buffer.
                               The first _mdSize_ bytes represent the metadata.
                               The remaining _fvSize_ bytes represent the feature vector. */
};

typedef struct br_universal_template *br_utemplate;
typedef const struct br_universal_template *br_const_utemplate;

/*!
 * \brief br_universal_template constructor.
 * \see br_free_utemplate
 */
BR_EXPORT br_utemplate br_new_utemplate(int32_t algorithmID, uint32_t frame, int32_t x, int32_t y, uint32_t width, uint32_t height, float confidence, uint32_t personID, const char *metadata, const char *featureVector, uint32_t fvSize);

/*!
 * \brief br_universal_template destructor.
 * \see br_new_utemplate
 */
BR_EXPORT void br_free_utemplate(br_const_utemplate utemplate);

/*!
 * \brief Serialize a br_universal_template to a file.
 * \see br_append_utemplate_contents
 */
BR_EXPORT void br_append_utemplate(FILE *file, br_const_utemplate utemplate);

/*!
 * \brief br_universal_template iterator callback.
 * \see br_iterate_utemplates
 */
typedef void *br_callback_context;
typedef void (*br_utemplate_callback)(br_const_utemplate, br_callback_context);

/*!
 * \brief Iterate over an inplace array of br_universal_template.
 * \see br_iterate_utemplates_file
 */
BR_EXPORT void br_iterate_utemplates(br_const_utemplate begin, br_const_utemplate end, br_utemplate_callback callback, br_callback_context context);

/*!
 * \brief Iterate over br_universal_template in a file.
 * \return The number of templates iterated
 * \see br_iterate_utemplates
 */
BR_EXPORT int br_iterate_utemplates_file(FILE *file, br_utemplate_callback callback, br_callback_context context, bool parallel);

/*!
 * \brief Write a message annotated with the current time to stderr.
 */
BR_EXPORT void br_log(const char *message);

#ifdef __cplusplus
}
#endif

#endif // BR_UNIVERSAL_TEMPLATE_H
