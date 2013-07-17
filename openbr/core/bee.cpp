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
        QStringList metadata;
        if (!ignoreMetadata)
            foreach (const QString &key, file.localKeys()) {
                if ((key == "Index") || (key == "Subject")) continue;
                metadata.append(key+"=\""+QtUtils::toString(file.value(key))+"\"");
            }
        lines.append("\t<biometric-signature name=\"" + file.get<QString>("Subject") +"\">");
        lines.append("\t\t<presentation file-name=\"" + file.name + "\" " + metadata.join(" ") + "/>");
        lines.append("\t</biometric-signature>");
    }
    lines.append("</biometric-signature-set>");
    QtUtils::writeFile(sigset, lines);
}

template <typename T>
Mat readMatrix(const br::File &matrix, QString *targetSigset = NULL, QString *querySigset = NULL)
{
    // Special case matrix construction
    if (matrix == "Identity") {
        int rows = matrix.get<int>("rows", -1);
        int columns = matrix.get<int>("columns", -1);
        const int size = matrix.get<int>("size", -1);
        if (size != -1) {
            if (rows == -1) rows = size;
            if (columns == -1) columns = size;
        }
        const int step = matrix.get<int>("step", 1);
        if (rows    % step != 0) qFatal("Step does not divide rows evenly.");
        if (columns % step != 0) qFatal("Step does not divide columns evenly.");

        if (sizeof(T) == sizeof(BEE::Mask_t)) {
            const bool selfSimilar = matrix.get<bool>("selfSimilar", false);

            Mat m(rows, columns, CV_8UC1);
            m.setTo(BEE::NonMatch);
            for (int i=0; i<std::min(rows, columns); i+=step)
                for (int j=0; j<step; j++)
                    for (int k=0; k<step; k++)
                        m.at<BEE::Mask_t>(i+j,i+k) = ((selfSimilar && (j == k)) ? BEE::DontCare : BEE::Match);
            return m;
        } else if (sizeof(T) == sizeof(BEE::Simmat_t)) {
            Mat m(rows, columns, CV_32FC1);
            m.setTo(0);
            for (int i=0; i<std::min(rows, columns); i+=step)
                for (int j=0; j<step; j++)
                    for (int k=0; k<step; k++)
                        m.at<BEE::Simmat_t>(i+j,i+k) = 1;
            return m;
        }
    }

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

    // Get matrix data
    qint64 bytesExpected = (qint64)rows*(qint64)cols*(qint64)sizeof(T);
    Mat m(rows, cols, OpenCVType<T,1>::make());
    if (file.read((char*)m.data, bytesExpected) != bytesExpected)
        qFatal("Invalid matrix size.");
    file.close();

    Mat result;
    if (isDistance ^ matrix.get<bool>("negate", false)) m.convertTo(result, -1, -1);
    else                                                result = m.clone();
    return result;
}

Mat BEE::readSimmat(const br::File &simmat, QString *targetSigset, QString *querySigset)
{
    return readMatrix<Simmat_t>(simmat, targetSigset, querySigset);
}

Mat BEE::readMask(const br::File &mask)
{
    return readMatrix<Mask_t>(mask);
}

template <typename T>
void writeMatrix(const Mat &m, const QString &matrix, const QString &targetSigset, const QString &querySigset)
{
    if (m.type() != OpenCVType<T,1>::make()) qFatal("Invalid matrix type.");

    int elemSize = sizeof(T);
    QString matrixType;
    if      (elemSize == 1) matrixType = "B";
    else if (elemSize == 4) matrixType = "F";
    else                    qFatal("Invalid element size.");

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

void BEE::writeSimmat(const Mat &m, const QString &simmat, const QString &targetSigset, const QString &querySigset)
{
    writeMatrix<Simmat_t>(m, simmat, targetSigset, querySigset);
}

void BEE::writeMask(const Mat &m, const QString &mask, const QString &targetSigset, const QString &querySigset)
{
    writeMatrix<Mask_t>(m, mask, targetSigset, querySigset);
}

void BEE::readMatrixHeader(const QString &matrix, QString *targetSigset, QString *querySigset)
{
    qDebug("Reading %s header.", qPrintable(matrix));
    if (matrix.endsWith("mask")) readMatrix<  Mask_t>(matrix, targetSigset, querySigset);
    else                         readMatrix<Simmat_t>(matrix, targetSigset, querySigset);
}

void BEE::writeMatrixHeader(const QString &matrix, const QString &targetSigset, const QString &querySigset)
{
    qDebug("Writing %s header to %s %s.", qPrintable(matrix), qPrintable(targetSigset), qPrintable(querySigset));
    if (matrix.endsWith("mask")) writeMatrix<  Mask_t>(readMatrix<  Mask_t>(matrix), matrix, targetSigset, querySigset);
    else                         writeMatrix<Simmat_t>(readMatrix<Simmat_t>(matrix), matrix, targetSigset, querySigset);
}

void BEE::makeMask(const QString &targetInput, const QString &queryInput, const QString &mask)
{
    qDebug("Making mask from %s and %s to %s", qPrintable(targetInput), qPrintable(queryInput), qPrintable(mask));
    FileList targets = TemplateList::fromGallery(targetInput).files();
    FileList queries = (queryInput == ".") ? targets : TemplateList::fromGallery(queryInput).files();
    int partitions = targets.first().get<int>("crossValidate");
    if (partitions == 0) writeMask(makeMask(targets, queries), mask, targetInput, queryInput);
    else {
        if (!mask.contains("%1")) qFatal("Mask file name missing partition number place marker (%%1)");
        for (int i=0; i<partitions; i++) {
            writeMask(makeMask(targets, queries, i), mask.arg(i), targetInput, queryInput);
        }
    }
}

cv::Mat BEE::makeMask(const br::FileList &targets, const br::FileList &queries, int partition)
{
    // Would like to use indexProperty for this, but didn't make a version of that for Filelist yet
    // -cao
    QList<QString> targetLabels = targets.get<QString>("Subject", "-1");
    QList<QString> queryLabels = queries.get<QString>("Subject", "-1");
    QList<int> targetPartitions = targets.crossValidationPartitions();
    QList<int> queryPartitions = queries.crossValidationPartitions();

    Mat mask(queries.size(), targets.size(), CV_8UC1);
    for (int i=0; i<queries.size(); i++) {
        const QString &fileA = queries[i];
        const QString labelA = queryLabels[i];
        const int partitionA = queryPartitions[i];

        for (int j=0; j<targets.size(); j++) {
            const QString &fileB = targets[j];
            const QString labelB = targetLabels[j];
            const int partitionB = targetPartitions[j];

            Mask_t val;
            if      (fileA == fileB)           val = DontCare;
            else if (labelA == "-1")             val = DontCare;
            else if (labelB == "-1")             val = DontCare;
            else if (partitionA != partition)  val = DontCare;
            else if (partitionB == -1)         val = NonMatch;
            else if (partitionB != partition)  val = DontCare;
            else if (partitionA != partitionB) val = DontCare;
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
        masks.append(readMask(inputMask));
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

    writeMask(combinedMask, outputMask, "Combined_Targets", "Combined_Queries");
}
