Constructor | Description
--- | ---
FileList() | Default constructor. Doesn't do anything.
FileList(int n) | Intialize the [FileList](filelist.md) with n empty [Files](../file/file.md)
FileList(const [QStringList][QStringList] &files) | Initialize the [FileList](filelist.md) with a list of strings. Each string should have the format "filename[key1=value1, key2=value2, ... keyN=valueN]"
FileList(const [QList][QList]&lt;[File](../file/file.md)&gt; &files) | Initialize the [FileList](filelist.md) from a list of [files](../file/file.md).

<!-- Links -->
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
