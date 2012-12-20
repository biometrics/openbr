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

#include <QIcon>
#include <QMetaProperty>

#include "transformeditor.h"
#include "transformlisteditor.h"

using namespace br;

br::TransformListEditor::TransformListEditor(const QList<Transform *> &transforms, const QString &separator, QWidget *parent)
    : QWidget(parent)
{
    this->separator = separator;

    foreach (br::Transform *transform, transforms) {
        parameters.append(new TransformEditor(transform, this));
        if (!separator.isEmpty() && (transform != transforms.last()))
            parameters.append(new QLabel(separator, this));
    }

    foreach (QWidget *parameter, parameters)
        layout.addWidget(parameter);

    addTransform.addMenu("Add...");
    //addTransform.setTitle("Add...");
    //addTransform.addAction("test");
    //addTransform.setIcon(QIcon("://glyphicons_190_circle_plus@2x.png"));
    layout.addWidget(&addTransform);
    setLayout(&layout);

    //connect(&addTransform, SIGNAL(clicked()), this, SLOT(addTransformClicked()));
}

void br::TransformListEditor::addTransformClicked()
{
    if (!separator.isEmpty()) {
        parameters.append(new QLabel(separator, this));
        layout.insertWidget(layout.count()-1, parameters.last());
    }
    parameters.append(new TransformEditor(Transform::make("Identity", this), this));
    layout.insertWidget(layout.count()-1, parameters.last());
}
