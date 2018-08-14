/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2017 Rank One Computing Corporation                             *
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

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treats the file as a JSON document.
 * \author Josh Klontz \cite jklontz
 */
class jsonGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(QString label READ get_label WRITE set_label RESET reset_label STORED false)
    Q_PROPERTY(QString filePath READ get_filePath WRITE set_filePath RESET reset_filePath STORED false)
    BR_PROPERTY(QString, label, "PersonID")
    BR_PROPERTY(QString, filePath, "Path")

    TemplateList readBlock(bool *done)
    {
        *done = true;

        if (!readOpen())
            qFatal("Failed to open JSON file!");

        QJsonParseError jsonParseError;
        const QJsonDocument jsonDocument = QJsonDocument::fromJson(f.readAll(), &jsonParseError);
        if (jsonParseError.error != QJsonParseError::NoError)
            qFatal("%s", qPrintable(jsonParseError.errorString()));

        if (!jsonDocument.isArray())
            qFatal("Expected JSON document to be an array!");

        TemplateList result;
        foreach (const QJsonValue &value, jsonDocument.array()) {
            File file(value.toObject().toVariantMap());
            if (!label.isEmpty() && file.contains(label))
                file.set("Label", file.get<QString>(label));
            if (!filePath.isEmpty() && file.contains(filePath))
                file.name = file.get<QString>(filePath);
            result.append(file);
        }

        return result;
    }

    void write(const Template &)
    {
        qFatal("Not implemented!");
    }
};

BR_REGISTER(Gallery, jsonGallery)

} // namespace br

#include "gallery/json.moc"
