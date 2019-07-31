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
#include <openbr/core/qtutils.h>
#include <openbr/core/common.h>

namespace br
{

struct CSVHeader
{
  QList<int> indices;

  CSVHeader()
  {}

  CSVHeader(const QString &key)
    : key(key)
  {}
  
  QString key;
  QStringList subKeys;  
};
  
class CSVHeaderList : public QList<CSVHeader>
{
public:
  CSVHeaderList()
  {}

  CSVHeaderList(const QList<CSVHeader> &headers)
  {
    foreach (const CSVHeader &header, headers)
      append(header);
  }

  CSVHeaderList(const QStringList &keys)
  {
    foreach (const QString &key, keys)
      append(CSVHeader(key));
  }
  
  void sort()
  {
    typedef QPair<QString, int> IndexPair;
    QList<IndexPair> sortedKeys = Common::Sort(keys());

    CSVHeaderList sortedList;
    foreach (const IndexPair sortedKey, sortedKeys)
      sortedList.append((*this)[sortedKey.second]);
    *this = sortedList;
  }

  QStringList keys() const
  {
    QStringList keys;
    for (int i=0; i<this->size(); i++)
      keys.append((*this)[i].key);
    return keys;
  }

    static CSVHeaderList fromHeaders(const QStringList &headers)
  {
    CSVHeaderList csvHeaders;
    QStringList processedKeys;
    
    for (int i=0; i<headers.size(); i++) {
      CSVHeader header;
      if (headers[i].contains("_")) {
	const QStringList subKeys = headers[i].split("_");
	header.key = subKeys.first();
	
	if (processedKeys.contains(header.key))
	  continue;
	else
	  processedKeys.append(header.key);
	
	header.subKeys.append(subKeys.last());
	header.indices.append(i);
	
	// Look for other subheaders with the same key
	for (int j=i+1; j<headers.size(); j++)
	  if (headers[j].contains("_")) {
	    const QStringList subKeys = headers[j].split("_");
	    if (subKeys.first() == header.key && !header.subKeys.contains(subKeys.last()) /* Check for ill-formed csvs */) {
	      header.indices.append(j);
	      header.subKeys.append(subKeys.last());
	    }
	  }
      } else {
	header.key = headers[i];
	header.indices.append(i);
      }
      csvHeaders.append(header);
    }
    return csvHeaders;
  }
};
  
/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 * \br_format Columns should be comma separated with first row containing headers.
 *            The first column in the file should be the path to the file to enroll.
 *            Other columns will be treated as file metadata.
 *
 * \br_related_plugin txtGallery
 */
class csvGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(bool inPlace READ get_inPlace WRITE set_inPlace RESET reset_inPlace STORED false)
    BR_PROPERTY(bool, inPlace, false)
    Q_PROPERTY(bool combineFiles READ get_combineFiles WRITE set_combineFiles RESET reset_combineFiles STORED false)
    BR_PROPERTY(bool, combineFiles, false)

    FileList files;
    CSVHeaderList headers;
  
    ~csvGallery()
    {
        f.close();
        if (files.isEmpty()) return;

        QSet<QString> samples;
        foreach (const File &file, files)
            foreach (const QString &key, file.localKeys())
                samples.insert(key);

        QStringList lines;
        lines.reserve(files.size()+1);

        // Make header
        headers = CSVHeaderList(samples.values());
        headers.sort();
        lines.append(QStringList(QStringList("File") + headers.keys()).join(","));

        // Make table
        foreach (const File &file, files)
            lines.append(lineFromFile(file));

        QtUtils::writeFile(file, lines);
    }

    void setValuesFromHeaders(File &f, const CSVHeaderList &headers, const QVariantList &values)
    {
        foreach (const CSVHeader &header, headers) {
            if (header.indices.size() == 1) {
                if (header.key == "Rects")
                    foreach(const QVariant &rect, values[header.indices.first()].toList())
                        f.appendRect(rect.toRectF());
                else if (header.key == "Points")
                    foreach(const QVariant &point, values[header.indices.first()].toList())
                        f.appendPoint(point.toPointF());
                else {
                    const QVariant value = values[header.indices.first()];
                    if (!value.canConvert<QString>() || !value.toString().isEmpty())
                        f.set(header.key, values[header.indices.first()]);
                }
            } else if (header.indices.size() == 2) { // QPointF
                const QPointF point(values[header.indices[header.subKeys.indexOf("X")]].toFloat(),
                                    values[header.indices[header.subKeys.indexOf("Y")]].toFloat());
                f.set(header.key, point);
                f.appendPoint(point);
            } else if (header.indices.size() == 4) { // QRectF
                const QRectF rect(values[header.indices[header.subKeys.indexOf("X")]].toFloat(),
                                  values[header.indices[header.subKeys.indexOf("Y")]].toFloat(),
                                  values[header.indices[header.subKeys.indexOf("Width")]].toFloat(),
                                  values[header.indices[header.subKeys.indexOf("Height")]].toFloat());
                f.set(header.key, rect);
                f.appendRect(rect);
            }
        }
    }
  
    TemplateList readBlock(bool *done)
    {
        readOpen();
        *done = false;
        TemplateList templates;
        if (!file.exists()) {
            *done = true;
            return templates;
        }

        if (f.pos() == 0) {
            // read header
            QByteArray lineBytes = f.readLine();
            QString line = QString::fromLocal8Bit(lineBytes).trimmed();
            QRegExp regexp("\\s*,\\s*");
            headers = CSVHeaderList::fromHeaders(line.split(regexp).mid(1));
        }

	if (combineFiles) {
  	    *done = true;
 	    QMap<QString, File> combinedFiles;
	  
  	    while (!f.atEnd()) {
                QVariantList values;
                foreach (const QString &value, QtUtils::parse(f.readLine(), ','))
                    values.append(QtUtils::fromString(value));

	       const QString name = values.first().toString();
	       File &in = combinedFiles[name];
	       in.name = name;
	       setValuesFromHeaders(in, headers, values.mid(1));
	    }

	    foreach (const File &in, combinedFiles.values())
	      templates.append(in);
	} else {
  	    for (qint64 i = 0; i < this->readBlockSize && !f.atEnd(); i++) {
                QVariantList values;
                foreach (const QString &value, QtUtils::parse(f.readLine(), ','))
                    values.append(QtUtils::fromString(value));

               File in;
	       in.name = values.first().toString();
	       setValuesFromHeaders(in, headers, values.mid(1));
	       in.set("progress", f.pos());
	       templates.append(in);
	    }
	    *done = f.atEnd();
        }
       
        return templates;
    }

    void write(const Template &t)
    {
        if (inPlace) {
            writeOpen();
            if (headers.isEmpty()) {
                foreach (const QString &key, t.file.localKeys())
		  headers.append(CSVHeader(key));

                headers.sort();
                const QString header = QString(QStringList(QStringList("File") + headers.keys()).join(",") + "\n");
                f.write(header.toLocal8Bit());
            }
            f.write(QString(lineFromFile(t.file) + "\n").toLocal8Bit());
        } else
            files.append(t.file);
    }

    QString lineFromFile(const br::File file)
    {
        QStringList words;
        words.append(file.name);
        foreach (const QString &key, headers.keys()) {
            QString value = QtUtils::toString(file.value(key));
            if (value.contains(","))
                value = '"' + value + '"';
            words.append(value);
        }
        return words.join(",");
    }
};

BR_REGISTER(Gallery, csvGallery)

} // namespace br

#include "gallery/csv.moc"
