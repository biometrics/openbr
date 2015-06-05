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

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \brief DOCUMENT ME
 * \author Unknown \cite unknown
 */
class EventTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString eventName READ get_eventName WRITE set_eventName RESET reset_eventName STORED false)
    BR_PROPERTY(QString, eventName, "")

    TemplateEvent event;

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        event.pulseSignal(dst);
    }

    TemplateEvent *getEvent(const QString &name)
    {
        return name == eventName ? &event : NULL;
    }
};

BR_REGISTER(Transform, EventTransform)

} // namespace br

#include "core/event.moc"
