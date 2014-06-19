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

/*!
 * \brief A flat template format for representing arbitrary feature vectors.
 */
struct br_universal_template
{
    unsigned char imageID[16];    /*!< MD5 hash of the undecoded origin file. */
    unsigned char templateID[16]; /*!< MD5 hash of _data_. */
    int32_t  algorithmID;    /*!< type of _data_. */
    uint32_t size;           /*!< length of _data_. */
    unsigned char data[];    /*!< _size_-byte buffer. */
};

typedef struct br_universal_template *br_utemplate;
typedef const struct br_universal_template *br_const_utemplate;

/*!
 * \brief br_universal_template constructor.
 * \see br_free_utemplate
 */
BR_EXPORT br_utemplate br_new_utemplate(const int8_t *imageID, const int8_t *templateID, int32_t algorithmID, uint32_t size, const int8_t *data);

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
 * \brief Serialize a br_universal_template to a file.
 * \see br_append_utemplate
 */
BR_EXPORT void br_append_utemplate_contents(FILE *file, const unsigned char *imageID, const unsigned char *templateID, int32_t algorithmID, uint32_t size, const unsigned char *data);

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
 * \see br_iterate_utemplates
 */
BR_EXPORT void br_iterate_utemplates_file(FILE *file, br_utemplate_callback callback, br_callback_context context, bool parallel);

#ifdef __cplusplus
}
#endif

#endif // BR_UNIVERSAL_TEMPLATE_H
