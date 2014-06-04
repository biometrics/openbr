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

#include <stdint.h>

/*!
 * \brief A flat template format for representing arbitrary feature vectors.
 */
struct br_universal_template
{
    uint8_t  imageID[16];    /*!< MD5 hash of the undecoded origin file. */
    uint8_t  templateID[16]; /*!< MD5 hash of _data_. */
    int32_t  algorithmID;    /*!< type of _data_. */
    uint32_t size;           /*!< length of _data_. */
    uint8_t  data[];         /*!< _size_-byte buffer. */
};

#endif // BR_UNIVERSAL_TEMPLATE_H
