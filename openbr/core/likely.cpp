/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
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

#include "likely.h"

using namespace br;

void br::Likely(const QString &inputType, const QString &outputType, const File &outputSourceFile)
{
    const QSharedPointer<Transform> t(Transform::fromAlgorithm(Globals->algorithm));

    QFile file(outputSourceFile.name);
    if (!file.open(QFile::WriteOnly))
        qFatal("Failed to open Likely output source file for writing!");

    file.write("; Automatically generated source code from:\n");
    file.write("; $ br -algorithm ");
    file.write(Globals->algorithm.toLatin1());
    file.write(" -likely ");
    file.write(inputType.toLatin1());
    file.write(" ");
    file.write(outputType.toLatin1());
    file.write(" ");
    file.write(outputSourceFile.flat().toLatin1());
    file.write("\n\n");

    file.write("f :=\n");
    file.write("  src :->\n");

    file.write(t->likely(""));
    file.write("\n");

    file.write("\n(extern ");
    file.write(outputType.toLatin1());
    file.write(" ");
    file.write(Globals->algorithm.toLower().toLatin1());
    file.write(" ");
    file.write(inputType.toLatin1());
    file.write(" f)\n");
}
