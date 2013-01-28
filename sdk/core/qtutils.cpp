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

#include <QCryptographicHash>
#include <QDebug>
#ifndef BR_EMBEDDED
#include <QDesktopServices>
#endif // BR_EMBEDDED
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRegExp>
#include <QStack>
#include <QUrl>
#include <openbr_plugin.h>

#include "qtutils.h"

using namespace br;

QStringList QtUtils::getFiles(QDir dir, bool recursive)
{
    dir = QDir(dir.canonicalPath());

    QStringList files;
    foreach (const QString &file, dir.entryList(QDir::Files))
        files.append(QDir::cleanPath(dir.absoluteFilePath(file)));

    if (!recursive) return files;

    foreach (const QString &folder, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        QDir subdir(dir);
        bool success = subdir.cd(folder); if (!success) qFatal("QtUtils::getFiles cd failure.");
        files.append(getFiles(subdir, true));
    }
    return files;
}

QStringList QtUtils::getFiles(const QString &regexp)
{
    QFileInfo fileInfo(regexp);
    QDir dir(fileInfo.dir());
    QRegExp re(fileInfo.fileName());
    re.setPatternSyntax(QRegExp::Wildcard);

    QStringList files;
    foreach (const QString &fileName, dir.entryList(QDir::Files))
        if (re.exactMatch(fileName))
            files.append(dir.filePath(fileName));
    return files;
}

QStringList QtUtils::readLines(const QString &file)
{
    QStringList lines;
    readFile(file, lines);
    return lines;
}

void QtUtils::readFile(const QString &file, QStringList &lines)
{
    QFile f(file);
    if (!f.open(QFile::ReadOnly)) qFatal("QtUtils::readFile unable to open %s for reading.", qPrintable(file));
    lines = QString(f.readAll()).split('\n', QString::SkipEmptyParts);
    for (int i=0; i<lines.size(); i++)
        lines[i] = lines[i].simplified();
    f.close();
}

void QtUtils::readFile(const QString &file, QByteArray &data, bool uncompress)
{
    QFile f(file);
    if (!f.open(QFile::ReadOnly)) qFatal("QtUtils::readFile unable to open %s for reading.", qPrintable(file));
    data = f.readAll();
    if (uncompress) data = qUncompress(data);
    f.close();
}

void QtUtils::writeFile(const QString &file, const QStringList &lines)
{
    writeFile(file, lines.join("\n") + "\n");
}

void QtUtils::writeFile(const QString &file, const QString &data)
{
    writeFile(file, data.toLocal8Bit());
}

void QtUtils::writeFile(const QString &file, const QByteArray &data, int compression)
{
    const QString baseName = QFileInfo(file).baseName();
    const QByteArray contents = (compression == 0) ? data : qCompress(data, compression);
    if (baseName == "terminal") {
        printf("%s", qPrintable(contents));
    } else if (baseName == "buffer") {
        Globals->buffer = data;
    } else {
        QFile f(file);
        touchDir(f);
        if (!f.open(QFile::WriteOnly))
            qFatal("QtUtils::writeFile failed to open %s for writing.", qPrintable(file));
        f.write(contents);
        f.close();
    }
}

void QtUtils::touchDir(const QDir &dir)
{
    if (dir.exists(".")) return;
    if (!dir.mkpath("."))
        qFatal("Unable to create path to dir %s", qPrintable(dir.absolutePath()));
}

void QtUtils::touchDir(const QFile &file)
{
    touchDir(QFileInfo(file));
}

void QtUtils::touchDir(const QFileInfo &fileInfo)
{
    touchDir(fileInfo.dir());
}

void QtUtils::emptyDir(QDir &dir)
{
    foreach (const QString &folder, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks)) {
        QDir subdir(dir);
        bool success = subdir.cd(folder); if (!success) qFatal("QtUtils::emptyDir cd failure.");
        emptyDir(subdir);
    }

    foreach (const QString &file, dir.entryList(QDir::Files))
        dir.remove(file);

    foreach (const QString &folder, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks))
        dir.rmdir(folder);

    foreach (const QString &symlink, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
        dir.remove(symlink);
}

void QtUtils::deleteDir(QDir &dir)
{
    emptyDir(dir);
    dir.rmdir(".");
}

QString QtUtils::find(const QString &file, const QString &alt)
{
    if (QFileInfo(file).exists()) return file;
    if (QFileInfo(alt).exists()) return alt;
    qFatal("QtUtils::find can't find file %s or alt %s\n", qPrintable(file), qPrintable(alt));
    return "";
}

bool QtUtils::toBool(const QString &string)
{
    bool ok;
    bool result = (bool)string.toInt(&ok); if (!ok) qFatal("QtUtils::toBool expected integer value, got %s.", qPrintable(string));
    return result;
}

int QtUtils::toInt(const QString &string)
{
    bool ok;
    int result = string.toInt(&ok); if (!ok) qFatal("QtUtils::toInt expected integer value, got %s.", qPrintable(string));
    return result;
}

float QtUtils::toFloat(const QString &string)
{
    bool ok;
    float result = string.toFloat(&ok); if (!ok) qFatal("QtUtils::toFloat expected floating point value, got %s.", qPrintable(string));
    return result;
}

QList<float> QtUtils::toFloats(const QStringList &strings)
{
    QList<float> floats;
    bool ok;
    foreach (const QString &string, strings) {
        floats.append(string.toFloat(&ok));
        if (!ok) qFatal("QtUtils::toFloats failed to convert %s to floating point format.", qPrintable(string));
    }
    return floats;
}

QStringList QtUtils::toStringList(const std::vector<std::string> &string_list)
{
    QStringList result;
    foreach (const std::string &string, string_list)
        result.append(QString::fromStdString(string));
    return result;
}

QStringList QtUtils::toStringList(int num_strings, const char *strings[])
{
    QStringList result;
    for (int i=0; i<num_strings; i++)
        result.append(strings[i]);
    return result;
}

QString QtUtils::shortTextHash(QString string)
{
    string.remove(QRegExp("[{}<>&]"));
    return QString(QCryptographicHash::hash(qPrintable(string), QCryptographicHash::Md5).toBase64()).remove(QRegExp("[^a-zA-Z1-9]")).left(6);
}

QStringList QtUtils::parse(QString args, char split)
{
    if (args.isEmpty()) return QStringList();

    QStringList words;
    int start = 0;
    bool inQuote = false;
    QStack<char> subexpressions;
    for (int i=0; i<args.size(); i++) {
        if (inQuote) {
            if (args[i] == '\'')
                inQuote = false;
        } else {
            if (args[i] == '\'') {
                inQuote = true;
            } else if ((args[i] == '(') || (args[i] == '[') || (args[i] == '<') || (args[i] == '{')) {
                subexpressions.push(args[i].toAscii());
            } else if (args[i] == ')') {
                if (subexpressions.pop() != '(') qFatal("QtUtils::parse unexpected ')'.");
            } else if (args[i] == ']') {
                if (subexpressions.pop() != '[') qFatal("QtUtils::parse unexpected ']'.");
            } else if (args[i] == '>') {
                if (subexpressions.pop() != '<') qFatal("QtUtils::parse unexpected '>'.");
            } else if (args[i] == '}') {
                if (subexpressions.pop() != '{') qFatal("QtUtils::parse unexpected '}'.");
            } else if (subexpressions.isEmpty() && (args[i] == split)) {
                words.append(args.mid(start, i-start).trimmed());
                start = i+1;
            }
        }
    }

    words.append(args.mid(start).trimmed());
    return words;
}

void QtUtils::checkArgsSize(const QString &name, const QStringList &args, int min, int max)
{
    if (max == -1) max = std::numeric_limits<int>::max();
    if (max == 0) max = min;
    if (args.size() < min) qFatal("%s expects at least %d arguments, got %d", qPrintable(name), min, args.size());
    if (args.size() > max) qFatal("%s expects no more than %d arguments, got %d", qPrintable(name), max, args.size());
}

bool QtUtils::runRScript(const QString &file)
{
    QProcess RScript;
    RScript.start("Rscript", QStringList() << file);
    RScript.waitForFinished(-1);
    bool result = ((RScript.exitCode() == 0) && (RScript.error() == QProcess::UnknownError));
    if (!result) qDebug("Failed to run 'Rscript', did you forget to install R?  See online documentation of 'br_plot' for required R packages.");
    return result;
}

bool QtUtils::runDot(const QString &file)
{
    QProcess dot;
    dot.start("dot -Tpdf -O " + file);
    dot.waitForFinished(-1);
    return ((dot.exitCode() == 0) && (dot.error() == QProcess::UnknownError));
}

void QtUtils::showFile(const QString &file)
{
#ifndef BR_EMBEDDED
    QDesktopServices::openUrl(QUrl("file:///" + QFileInfo(file).absoluteFilePath()));
#else // BR_EMBEDDED
    (void) file;
#endif // BR_EMBEDDED
}
