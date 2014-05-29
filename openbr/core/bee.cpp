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

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QMap>
#include <QRegExp>
#ifndef BR_EMBEDDED
#include <QtXml>
#endif // BR_EMBEDDED
#include <algorithm>
#include <limits>
#include <openbr/openbr_plugin.h>

#include "bee.h"
#include "opencvutils.h"
#include "qtutils.h"

using namespace cv;
using namespace br;

/**** BEE ****/
FileList BEE::readSigset(const File &sigset, bool ignoreMetadata)
{
    FileList fileList;

#ifndef BR_EMBEDDED
    QDomDocument doc(sigset.fileName());
    QFile file(sigset.resolved());
    bool success;
    success = file.open(QIODevice::ReadOnly); if (!success) qFatal("Unable to open %s for reading.", qPrintable(sigset));
    success = doc.setContent(&file);

    file.close();

    if (!success) {
        qWarning("Unable to parse %s.", qPrintable(sigset));
        return fileList;
    }

    QDomElement docElem = doc.documentElement();
    if (docElem.nodeName() != "biometric-signature-set")
        return fileList;

    QDomNode subject = docElem.firstChild();
    while (!subject.isNull()) {
        // Looping through subjects
        QDomNode fileNode = subject.firstChild();
        QDomElement d = subject.toElement();
        QString name = d.attribute("name");
        while (!fileNode.isNull()) {
            // Looping through files
            File file("", name);

            QDomElement e = fileNode.toElement();
            QDomNamedNodeMap attributes = e.attributes();
            for (int i=0; i<attributes.length(); i++) {
                const QString key = attributes.item(i).nodeName();
                const QString value = attributes.item(i).nodeValue();
                if      (key == "file-name") file.name = value;
                else if (!ignoreMetadata)    file.set(key, value);
            }

            // add bounding boxes, if they exist (will be child elements of <presentation>)
            if (fileNode.hasChildNodes()) {
                QList<QRectF> rects;
                QDomNodeList bboxes = fileNode.childNodes();
                for (int i=0; i<bboxes.length(); i++) {
                    QDomElement bbox = bboxes.at(i).toElement();
                    qreal x = bbox.attribute("x").toDouble();
                    qreal y = bbox.attribute("y").toDouble();
                    qreal width = bbox.attribute("width").toDouble();
                    qreal height = bbox.attribute("height").toDouble();
                    rects += QRectF(x, y, width, height);
                }
                file.setRects(rects);
            }

            if (file.name.isEmpty()) qFatal("Missing file-name in %s.", qPrintable(sigset));
            fileList.append(file);

            fileNode = fileNode.nextSibling();
        }
        subject = subject.nextSibling();
    }
#else // BR_EMBEDDED
    (void) sigset;
    (void) ignoreMetadata;
#endif // BR_EMBEDDED

    return fileList;
}

void BEE::writeSigset(const QString &sigset, const br::FileList &files, bool ignoreMetadata)
{
    QStringList lines; lines.reserve(3*files.size()+3);
    lines.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    lines.append("<biometric-signature-set>");
    foreach (const File &file, files) {
        if (!file.isNull()) {
            QStringList metadata;
            if (!ignoreMetadata) {
                foreach (const QString &key, file.localKeys()) {
                    if ((key == "Index") || (key == "Label") || (key == "Points") || (key == "Rects")) continue;
                    metadata.append(key+"=\""+QtUtils::toString(file.value(key))+"\"");
                }
                QStringList landmarks;
                if (!file.points().isEmpty()) {
                    foreach (const QPointF &point, file.points()) landmarks.append(QtUtils::toString(point));
                    metadata.append("Points=\"["+landmarks.join(",")+"]\""); landmarks.clear();
                }
                if (!file.rects().isEmpty()) {
                    foreach (const QRectF &rect, file.rects()) landmarks.append(QtUtils::toString(rect));
                    metadata.append("Rects=\"["+landmarks.join(",")+"]\"");
                }
            }
            lines.append("\t<biometric-signature name=\"" + file.get<QString>("Label",file.baseName()) +"\">");
            lines.append("\t\t<presentation file-name=\"" + file.name + "\" " + metadata.join(" ") + "/>");
            lines.append("\t</biometric-signature>");
        }
    }
    lines.append("</biometric-signature-set>");
    QtUtils::writeFile(sigset, lines);
}

Mat BEE::readMat(const br::File &matrix, QString *targetSigset, QString *querySigset)
{
    QFile file(matrix);
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("Unable to open %s for reading.", qPrintable(matrix.name));

    // Check format
    QByteArray format = file.readLine();
    bool isDistance = (format[0] == 'D');
    if (format[1] != '2') qFatal("Invalid matrix header.");


    // Read sigsets
    if (targetSigset != NULL) *targetSigset = file.readLine().simplified();
    else                      file.readLine();
    if (querySigset != NULL) *querySigset = file.readLine().simplified();
    else                     file.readLine();

    // Get matrix size
    QStringList words = QString(file.readLine()).split(" ");
    int rows = words[1].toInt();
    int cols = words[2].toInt();

    bool isMask = words[0][1] == 'B';
    int typeSize = isMask ? sizeof(BEE::Mask_t) : sizeof(BEE::Simmat_t);

    // Get matrix data
    Mat m;
    if (isMask)
        m.create(rows, cols, OpenCVType<BEE::Mask_t,1>::make());
    else
        m.create(rows, cols, OpenCVType<BEE::Simmat_t,1>::make());

    qint64 bytesPerRow = m.cols * typeSize;

    for (int i=0; i < m.rows;i++)
    {
        cv::Mat aRow = m.row(i);
        qint64 bytesRead = file.read((char *)aRow.data, bytesPerRow);
        if (bytesRead != bytesPerRow)
        {
            qFatal("Didn't read complete row!");
        }
    }
    if (!file.atEnd())
        qFatal("Expected matrix end of file.");
    file.close();

    Mat result = m;
    if (isDistance ^ matrix.get<bool>("negate", false)) m.convertTo(result, -1, -1);

    return result;
}

void BEE::writeMat(const Mat &m, const QString &matrix, const QString &targetSigset, const QString &querySigset)
{
    bool isMask = false;
    if (m.type() == OpenCVType<BEE::Mask_t,1>::make())
        isMask = true;
    else if (m.type() != OpenCVType<BEE::Simmat_t,1>::make())
        qFatal("Invalid matrix type, .mtx files can only contain single channel float or uchar matrices.");

    int elemSize = isMask ? sizeof(BEE::Mask_t) : sizeof(BEE::Simmat_t);

    QString matrixType = isMask ? "B" : "F";

    char buff[4];
    QFile file(matrix);
    QtUtils::touchDir(file);
    bool success = file.open(QFile::WriteOnly); if (!success) qFatal("Unable to open %s for writing.", qPrintable(matrix));
    file.write("S2\n");
    file.write(qPrintable(targetSigset));
    file.write("\n");
    file.write(qPrintable(querySigset));
    file.write("\n");
    file.write("M");
    file.write(qPrintable(matrixType));
    file.write(" ");
    file.write(qPrintable(QString::number(m.rows)));
    file.write(" ");
    file.write(qPrintable(QString::number(m.cols)));
    file.write(" ");
    int endian = 0x12345678;
    memcpy(&buff, &endian, 4);
    file.write(buff, 4);
    file.write("\n");
    file.write((const char*)m.data, m.rows*m.cols*elemSize);
    file.close();
}

void BEE::readMatrixHeader(const QString &matrix, QString *targetSigset, QString *querySigset)
{
    qDebug("Reading %s header.", qPrintable(matrix));
    readMat(matrix, targetSigset, querySigset);
}

void BEE::writeMatrixHeader(const QString &matrix, const QString &targetSigset, const QString &querySigset)
{
    qDebug("Writing %s header to %s %s.", qPrintable(matrix), qPrintable(targetSigset), qPrintable(querySigset));

    writeMat(readMat(matrix), matrix, targetSigset, querySigset);
}

void BEE::makeMask(const QString &targetInput, const QString &queryInput, const QString &mask)
{
    qDebug("Making mask from %s and %s to %s", qPrintable(targetInput), qPrintable(queryInput), qPrintable(mask));
    FileList targets = TemplateList::fromGallery(targetInput).files();
    FileList queries = (queryInput == ".") ? targets : TemplateList::fromGallery(queryInput).files();
    int partitions = targets.first().get<int>("crossValidate");
    if (partitions == 0) writeMat(makeMask(targets, queries), mask, targetInput, queryInput);
    else {
        if (!mask.contains("%1")) qFatal("Mask file name missing partition number place marker (%%1)");
        for (int i=0; i<partitions; i++) {
            writeMat(makeMask(targets, queries, i), mask.arg(i), targetInput, queryInput);
        }
    }
}

void BEE::makePairwiseMask(const QString &targetInput, const QString &queryInput, const QString &mask)
{
    FileList targets = TemplateList::fromGallery(targetInput).files();
    FileList queries = (queryInput == ".") ? targets : TemplateList::fromGallery(queryInput).files();
    int partitions = targets.first().get<int>("crossValidate");
    if (partitions == 0) writeMat(makePairwiseMask(targets, queries), mask, targetInput, queryInput);
    else {
        if (!mask.contains("%1")) qFatal("Mask file name missing partition number place marker (%%1)");
        for (int i=0; i<partitions; i++) {
            writeMat(makePairwiseMask(targets, queries, i), mask.arg(i), targetInput, queryInput);
        }
    }
}

cv::Mat BEE::makePairwiseMask(const br::FileList &targets, const br::FileList &queries, int partition)
{
    // Direct use of "Label" isn't general, also would prefer to use indexProperty, rather than
    // doing string comparisons (but that isn't implemented yet for FileList) -cao
    QList<QString> targetLabels = File::get<QString>(targets, "Label", "-1");
    QList<QString> queryLabels = File::get<QString>(queries, "Label", "-1");

    QList<int> targetPartitions = targets.crossValidationPartitions();
    QList<int> queryPartitions = queries.crossValidationPartitions();

    Mat mask(queries.size(), 1, CV_8UC1);
    for (int i=0; i<queries.size(); i++) {
        const QString &fileA = queries[i];
        const QString labelA = queryLabels[i];
        const int partitionA = queryPartitions[i];

        const QString &fileB = targets[i];
        const QString labelB = targetLabels[i];
        const int partitionB = targetPartitions[i];

        Mask_t val;
        if      (fileA == fileB)           val = DontCare;
        else if (labelA == "-1")           val = DontCare;
        else if (labelB == "-1")           val = DontCare;
        else if (partitionA != partition)  val = DontCare;
        else if (partitionB == -1)         val = NonMatch;
        else if (partitionB != partition)  val = DontCare;
        else if (labelA == labelB)         val = Match;
        else                               val = NonMatch;
        mask.at<Mask_t>(i,0) = val;
    }

    return mask;
}

cv::Mat BEE::makeMask(const br::FileList &targets, const br::FileList &queries, int partition)
{
    // Direct use of "Label" isn't general, also would prefer to use indexProperty, rather than
    // doing string comparisons (but that isn't implemented yet for FileList) -cao
    QList<QString> targetLabels = File::get<QString>(targets, "Label", "-1");
    QList<QString> queryLabels = File::get<QString>(queries, "Label", "-1");

    QList<int> targetPartitions = targets.crossValidationPartitions();
    QList<int> queryPartitions = queries.crossValidationPartitions();

    QList<bool> targetsOnly = File::get<bool>(queries, "targetOnly", false);

    Mat mask(queries.size(), targets.size(), CV_8UC1);
    for (int i=0; i<queries.size(); i++) {
        const QString &fileA = queries[i];
        const QString labelA = queryLabels[i];
        const int partitionA = queryPartitions[i];
        const bool targetOnly = targetsOnly[i];

        for (int j=0; j<targets.size(); j++) {
            const QString &fileB = targets[j];
            const QString labelB = targetLabels[j];
            const int partitionB = targetPartitions[j];

            Mask_t val;
            if      (fileA == fileB)           val = DontCare;
            else if (targetOnly)               val = DontCare;
            else if (labelA == "-1")           val = DontCare;
            else if (labelB == "-1")           val = DontCare;
            else if (partitionA != partition)  val = DontCare;
            else if (partitionB == -1)         val = NonMatch;
            else if (partitionB != partition)  val = DontCare;
            else if (labelA == labelB)         val = Match;
            else                               val = NonMatch;
            mask.at<Mask_t>(i,j) = val;
        }
    }

    return mask;
}

void BEE::combineMasks(const QStringList &inputMasks, const QString &outputMask, const QString &method)
{
    qDebug("Combining %d masks to %s with method %s", inputMasks.size(), qPrintable(outputMask), qPrintable(method));

    bool AND = true;
    if      (method == "And") AND = true;
    else if (method == "Or")  AND = false;
    else                      qFatal("Invalid method.");

    QList<Mat> masks;
    foreach (const QString &inputMask, inputMasks)
        masks.append(readMat(inputMask));
    if (masks.size() < 2) qFatal("Expected at least two masks.");

    const int rows = masks.first().rows;
    const int columns = masks.first().cols;

    Mat combinedMask(rows, columns, CV_8UC1);
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            int genuineCount = 0;
            int imposterCount = 0;
            int dontcareCount = 0;
            for (int k=0; k<masks.size(); k++) {
                switch (masks[k].at<Mask_t>(i,j)) {
                  case Match:
                    genuineCount++;
                    break;
                  case NonMatch:
                    imposterCount++;
                    break;
                  case DontCare:
                    dontcareCount++;
                    break;
                }
            }
            if ((genuineCount != 0) && (imposterCount != 0)) qFatal("Comparison is both a genuine and an imposter.");

            Mask_t val;
            if      (genuineCount > 0)  val = Match;
            else if (imposterCount > 0) val = NonMatch;
            else                        val = DontCare;
            if (AND && (dontcareCount > 0)) val = DontCare;
            combinedMask.at<Mask_t>(i,j) = val;
        }
    }

    writeMat(combinedMask, outputMask, "Combined_Targets", "Combined_Queries");
}
