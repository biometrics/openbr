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

#ifndef BR_TRANSFORMEDITOR_H
#define BR_TRANSFORMEDITOR_H

#include <QHBoxLayout>
#include <QComboBox>
#include <QWidget>
#include <openbr/openbr_plugin.h>

namespace br
{

class BR_EXPORT TransformEditor : public QWidget
{
    Q_OBJECT
    QHBoxLayout layout;
    QComboBox name;
    QList<QWidget*> parameters;

public:
    explicit TransformEditor(br::Transform *transform, QWidget *parent = 0);
};

} // namespace br

#endif // BR_TRANSFORMEDITOR_H
