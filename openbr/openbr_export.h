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

#ifndef OPENBR_EXPORT_H
#define OPENBR_EXPORT_H

#if defined BR_EMBEDDED
#  define BR_EXPORT
#else
#  if defined BR_LIBRARY
#    if defined _WIN32 || defined __CYGWIN__
#      define BR_EXPORT __declspec(dllexport)
#    else
#      define BR_EXPORT __attribute__((visibility("default")))
#    endif
#  else
#    if defined _WIN32 || defined __CYGWIN__
#      define BR_EXPORT __declspec(dllimport)
#    else
#      define BR_EXPORT
#    endif
#  endif
#endif

#if defined BR_LIBRARY
#  if defined _WIN32 || defined __CYGWIN__
#    define BR_EXPORT_ALWAYS __declspec(dllexport)
#  else
#    define BR_EXPORT_ALWAYS __attribute__((visibility("default")))
#  endif
#else
#  if defined _WIN32 || defined __CYGWIN__
#    define BR_EXPORT_ALWAYS __declspec(dllimport)
#  else
#    define BR_EXPORT_ALWAYS
#  endif
#endif

#endif // OPENBR_EXPORT_H
