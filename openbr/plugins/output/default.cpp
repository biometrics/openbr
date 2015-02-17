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
 * \ingroup outputs
 * \brief Adaptor class -- write a matrix output using Format classes.
 * \author Charles Otto \cite caotto
 */
class DefaultOutput : public MatrixOutput
{
    Q_OBJECT

    ~DefaultOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        br::Template T(this->file, this->data);
        QScopedPointer<Format> writer(Factory<Format>::make(this->file));
        writer->write(T);
    }
};

BR_REGISTER(Output, DefaultOutput)

} // namespace br

#include "output/default.moc"
