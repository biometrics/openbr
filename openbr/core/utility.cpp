#include <openbr/core/qtutils.h>
#include "utility.h"
#include <QRegExp>

QStringList br::getFiles(QDir dir, bool recursive)
{
    dir = QDir(dir.canonicalPath());

    QStringList files;
    foreach (const QString &file, QtUtils::naturalSort(dir.entryList(QDir::Files)))
        files.append(dir.absoluteFilePath(file));

    if (!recursive) return files;

    foreach (const QString &folder, QtUtils::naturalSort(dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))) {
        QDir subdir(dir);
        bool success = subdir.cd(folder); if (!success) qFatal("cd failure.");
        files.append(getFiles(subdir, true));
    }
    return files;
}

QList<br::FilesWithLabel> br::getFilesWithLabels(QDir dir)
{
    dir = QDir(dir.canonicalPath());

    QStringList files;
    foreach (const QString &file, dir.entryList(QDir::Files))
        files.append(dir.absoluteFilePath(file));

    QList<br::FilesWithLabel> filesWithLabels;
    filesWithLabels.append(br::FilesWithLabel(dir.dirName(),files));

    foreach (const QString &folder, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        QDir subdir(dir);
        bool success = subdir.cd(folder); if (!success) qFatal("cd failure.");
        filesWithLabels.append(getFilesWithLabels(subdir));
    }
    return filesWithLabels;
}

QStringList br::getFiles(const QString &regexp)
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
