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

#include <QRegularExpression>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.
 * \author Charles Otto \cite caotto
 */
class RegexPropertyTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    Q_PROPERTY(QString inputProperty READ get_inputProperty WRITE set_inputProperty RESET reset_inputProperty STORED false)
    Q_PROPERTY(QString outputProperty READ get_outputProperty WRITE set_outputProperty RESET reset_outputProperty STORED false)
    BR_PROPERTY(QString, regexp, "(.*)")
    BR_PROPERTY(QString, inputProperty, "name")
    BR_PROPERTY(QString, outputProperty, "Label")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QRegularExpression re(regexp);
        QRegularExpressionMatch match = re.match(dst.get<QString>(inputProperty));
        if (!match.hasMatch())
            qFatal("Unable to match regular expression \"%s\" to base name \"%s\"!", qPrintable(regexp), qPrintable(dst.get<QString>(inputProperty)));
        dst.set(outputProperty, match.captured(match.lastCapturedIndex()));
    }
};

BR_REGISTER(Transform, RegexPropertyTransform)

} // namespace br

#include "metadata/regexproperty.moc"
