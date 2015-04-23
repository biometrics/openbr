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

#include <QtEndian>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads FBI EBTS transactions.
 * \author Scott Klum \cite sklum
 * \br_link https://www.fbibiospecs.org/ebts.html
 */
class ebtsFormat : public Format
{
    Q_OBJECT

    struct Field {
        int type;
        QList<QByteArray> data;
    };

    struct Record {
        int type;
        quint32 bytes;
        int position; // Starting position of record

        QHash<int,QList<QByteArray> > fields;
    };

    quint32 recordBytes(const QByteArray &byteArray, const float recordType, int from) const
    {
        bool ok;
        quint32 size;

        if (recordType == 4 || recordType == 7) {
            // read first four bytes
            ok = true;
            size = qFromBigEndian<quint32>((const uchar*)byteArray.mid(from,4).constData());
        } else {
            int index = byteArray.indexOf(QChar(0x1D), from);
            size = byteArray.mid(from, index-from).split(':').last().toInt(&ok);
        }

        return ok ? size : -1;
    }

    void parseRecord(const QByteArray &byteArray, Record &record) const
    {
        if (record.type == 4 || record.type == 7) {
            // Just a binary blob
            // Read everything after the first four bytes
            // Not current supported
        } else {
            // Continue reading fields until we get all the data
            unsigned int position = record.position;
            while (position < record.position + record.bytes) {
                int index = byteArray.indexOf(QChar(0x1D), position);
                Field field = parseField(byteArray.mid(position, index-position),QChar(0x1F));
                if (field.type == 999 ) {
                    // Data begin after the field identifier and the colon
                    int dataBegin = byteArray.indexOf(':', position)+1;
                    field.data.clear();
                    field.data.append(byteArray.mid(dataBegin, record.bytes-(dataBegin-record.position)));

                    // Data fields are always last in the record
                    record.fields.insert(field.type,field.data);
                    break;
                }
                // Advance the position accounting for the separator
                position += index-position+1;
                record.fields.insert(field.type,field.data);
            }
        }
    }

    Field parseField(const QByteArray &byteArray, const QChar &sep) const
    {
        bool ok;
        Field f;

        QList<QByteArray> data = byteArray.split(':');

        f.type = data.first().split('.').last().toInt(&ok);
        f.data = data.last().split(sep.toLatin1());

        return f;
    }

    Template read() const
    {
        QByteArray byteArray;
        QtUtils::readFile(file, byteArray);

        Template t;

        Mat m;

        QList<Record> records;

        // Read the type one record (every EBTS file will have one of these)
        Record r1;
        r1.type = 1;
        r1.position = 0;
        r1.bytes = recordBytes(byteArray,r1.type,r1.position);

        // The fields in a type 1 record are strictly defined
        QList<QByteArray> data = byteArray.mid(r1.position,r1.bytes).split(QChar(0x1D).toLatin1());
        foreach (const QByteArray &datum, data) {
            Field f = parseField(datum,QChar(0x1F));
            r1.fields.insert(f.type,f.data);
        }

        records.append(r1);

        // Read the type two record (every EBTS file will have one of these)
        Record r2;
        r2.type = 2;
        r2.position = r1.bytes;
        r2.bytes = recordBytes(byteArray,r2.type,r2.position);

        // The fields in a type 2 record are strictly defined
        data = byteArray.mid(r2.position,r2.bytes).split(QChar(0x1D).toLatin1());
        foreach (const QByteArray &datum, data) {
            Field f = parseField(datum,QChar(0x1F));
            r2.fields.insert(f.type,f.data);
        }

        // Demographics
        if (r2.fields.contains(18)) {
            QString name = r2.fields.value(18).first();
            QStringList names = name.split(',');
            t.file.set("FIRSTNAME", names.at(1));
            t.file.set("LASTNAME", names.at(0));
        }

        if (r2.fields.contains(22)) t.file.set("DOB", r2.fields.value(22).first().toInt());
        if (r2.fields.contains(24)) t.file.set("GENDER", QString(r2.fields.value(24).first()));
        if (r2.fields.contains(25)) t.file.set("RACE", QString(r2.fields.value(25).first()));

        if (t.file.contains("DOB")) {
            const QDate dob = QDate::fromString(t.file.get<QString>("DOB"), "yyyyMMdd");
            const QDate current = QDate::currentDate();
            int age = current.year() - dob.year();
            if (current.month() < dob.month()) age--;
            t.file.set("Age", age);
        }

        records.append(r2);

        // The third field of the first record contains informations about all the remaining records in the transaction
        // We don't care about the first two and the final items
        QList<QByteArray> recordTypes = r1.fields.value(3);
        for (int i=2; i<recordTypes.size()-1; i++) {
            // The first two bytes indicate the record index (and we don't want the separator), but we only care about the type
            QByteArray recordType = recordTypes[i].mid(3);
            Record r;
            r.type = recordType.toInt();
            records.append(r);
        }

        QList<int> frontalIdxs;
        int position = r1.bytes + r2.bytes;
        for (int i=2; i<records.size(); i++) {
            records[i].position = position;
            records[i].bytes = recordBytes(byteArray,records[i].type,position);

            parseRecord(byteArray, records[i]);
            if (records[i].type == 10) frontalIdxs.append(i);
            position += records[i].bytes;
        }

        if (!frontalIdxs.isEmpty()) {
            // We use the first type 10 record to get the frontal
            QByteArray frontal = records[frontalIdxs.first()].fields.value(999).first();
            m = imdecode(Mat(3, frontal.size(), CV_8UC3, frontal.data()), CV_LOAD_IMAGE_COLOR);
            if (!m.data) qWarning("ebtsFormat::read failed to decode image data.");
            t.m() = m;
        } else qWarning("ebtsFormat::cannot find image data within file.");

        return t;
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Writing EBTS files is not supported.");
    }
};

BR_REGISTER(Format, ebtsFormat)

} // namespace br

#include "format/ebts.moc"
