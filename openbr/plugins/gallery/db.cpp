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

#include <QtSql>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Database input.
 * \author Josh Klontz \cite jklontz
 */
class dbGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        br::File import = file.get<QString>("import", "");
        QString query = file.get<QString>("query");
        QString subset = file.get<QString>("subset", "");

        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
        db.setDatabaseName(file);
        if (!db.open()) qFatal("Failed to open SQLite database %s.", qPrintable(file.name));

        if (!import.isNull()) {
            qDebug("Parsing %s", qPrintable(import.name));
            QStringList lines = QtUtils::readLines(import);
            QList<QStringList> cells; cells.reserve(lines.size());
            const QRegExp re("\\s*,\\s*");
            foreach (const QString &line, lines) {
                cells.append(line.split(re));
                if (cells.last().size() != cells.first().size()) qFatal("Column count mismatch.");
            }

            QStringList columns, qMarks;
            QList<QVariantList> variantLists;
            for (int i=0; i<cells[0].size(); i++) {
                bool isNumeric;
                cells[1][i].toInt(&isNumeric);
                columns.append(cells[0][i] + (isNumeric ? " INTEGER" : " STRING"));
                qMarks.append("?");

                QVariantList variantList; variantList.reserve(lines.size()-1);
                for (int j=1; j<lines.size(); j++) {
                    if (isNumeric) variantList << cells[j][i].toInt();
                    else           variantList << cells[j][i];
                }
                variantLists.append(variantList);
            }

            const QString &table = import.baseName();
            qDebug("Creating table %s", qPrintable(table));
            QSqlQuery q(db);
            if (!q.exec("CREATE TABLE " + table + " (" + columns.join(", ") + ");"))
                qFatal("%s.", qPrintable(q.lastError().text()));
            if (!q.prepare("insert into " + table + " values (" + qMarks.join(", ") + ")"))
                qFatal("%s.", qPrintable(q.lastError().text()));
            foreach (const QVariantList &vl, variantLists)
                q.addBindValue(vl);
            if (!q.execBatch()) qFatal("%s.", qPrintable(q.lastError().text()));
        }

        QSqlQuery q(db);
        if (query.startsWith('\'') && query.endsWith('\''))
            query = query.mid(1, query.size()-2);
        if (!q.exec(query))
            qFatal("%s.", qPrintable(q.lastError().text()));

        if ((q.record().count() == 0) || (q.record().count() > 3))
            qFatal("Query record expected one to three fields, got %d.", q.record().count());
        const bool hasMetadata = (q.record().count() >= 2);
        const bool hasFilter = (q.record().count() >= 3);

        QString labelName = "Label";
        if (q.record().count() >= 2)
            labelName = q.record().fieldName(1);

        // subset = seed:subjectMaxSize:numSubjects:subjectMinSize or
        // subset = seed:{Metadata,...,Metadata}:numSubjects
        int seed = 0, subjectMaxSize = std::numeric_limits<int>::max(), numSubjects = std::numeric_limits<int>::max(), subjectMinSize = 0;
        QList<QRegExp> metadataFields;
        if (!subset.isEmpty()) {
            const QStringList &words = subset.split(":");
            QtUtils::checkArgsSize("Input", words, 2, 4);
            if      (words[0] == "train") seed = 0;
            else if (words[0] == "test" ) seed = 1;
            else                          seed = QtUtils::toInt(words[0]);
            if (words[1].startsWith('{') && words[1].endsWith('}')) {
                foreach (const QString &regexp, words[1].mid(1, words[1].size()-2).split(","))
                    metadataFields.append(QRegExp(regexp));
                subjectMaxSize = metadataFields.size();
            } else {
                subjectMaxSize = QtUtils::toInt(words[1]);
            }
            numSubjects = words.size() >= 3 ? QtUtils::toInt(words[2]) : std::numeric_limits<int>::max();
            subjectMinSize = words.size() >= 4 ? QtUtils::toInt(words[3]) : subjectMaxSize;
        }

        srand(seed);

        typedef QPair<QString,QString> Entry; // QPair<File,Metadata>
        QHash<QString, QList<Entry> > entries; // QHash<Label, QList<Entry> >
        while (q.next()) {
            if (hasFilter && (seed >= 0) && (qHash(q.value(2).toString()) % 2 != (uint)seed % 2)) continue; // Ensures training and testing filters don't overlap

            if (metadataFields.isEmpty())
                entries[hasMetadata ? q.value(1).toString() : ""].append(QPair<QString,QString>(q.value(0).toString(), hasFilter ? q.value(2).toString() : ""));
            else
                entries[hasFilter ? q.value(2).toString() : ""].append(QPair<QString,QString>(q.value(0).toString(), hasMetadata ? q.value(1).toString() : ""));
        }

        QStringList labels = entries.keys();
        qSort(labels);

        if (hasFilter && ((labels.size() > numSubjects) || (numSubjects == std::numeric_limits<int>::max())))
            std::random_shuffle(labels.begin(), labels.end());

        foreach (const QString &label, labels) {
            QList<Entry> entryList = entries[label];
            if ((entryList.size() >= subjectMinSize) && (numSubjects > 0)) {

                if (!metadataFields.isEmpty()) {
                    QList<Entry> subEntryList;
                    foreach (const QRegExp &metadata, metadataFields) {
                        for (int i=0; i<entryList.size(); i++) {
                            if (metadata.exactMatch(entryList[i].second)) {
                                subEntryList.append(entryList.takeAt(i));
                                break;
                            }
                        }
                    }
                    if (subEntryList.size() == metadataFields.size())
                        entryList = subEntryList;
                    else
                        continue;
                }

                if (entryList.size() > subjectMaxSize)
                    std::random_shuffle(entryList.begin(), entryList.end());
                foreach (const Entry &entry, entryList.mid(0, subjectMaxSize)) {
                    templates.append(File(entry.first));
                    templates.last().file.set(labelName, label);
                }
                numSubjects--;
            }
        }

        db.close();

        *done = true;
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not supported.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, dbGallery)

} // namespace br

#include "gallery/db.moc"
