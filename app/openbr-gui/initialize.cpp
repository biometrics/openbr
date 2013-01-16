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

#include <QDesktopServices>
#include <QMetaType>
#include <openbr_plugin.h>

#include "initialize.h"

void br_initialize_gui(const char *sdk_path)
{
    Q_INIT_RESOURCE(icons);
    qRegisterMetaType<br::File>("br::File");
    qRegisterMetaType<br::FileList>("br::FileList");
    qRegisterMetaType<br::Template>("br::Template");
    qRegisterMetaType<br::TemplateList>("br::TemplateList");
    br_initialize_qt(sdk_path);
    br_set_property("log", qPrintable(QString("%1/log.txt").arg(br_scratch_path())));
}
