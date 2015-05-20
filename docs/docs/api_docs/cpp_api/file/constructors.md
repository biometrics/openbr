Constructor | Description
--- | ---
File() | Default constructor. Sets [name](members.md#name) to false.
File(const [QString][QString] &file) | Initializes the file by calling the private function init.
File(const [QString][QString] &file, const [QVariant][QVariant] &label) | Initializes the file by calling the private function init. Append label to the [metadata](members.md#m_metadata) using the key "Label".
File(const char \*file) | Initializes the file with a c-style string.
File(const [QVariantMap][QVariantMap] &metadata) | Sets [name](members.md#name) to false and sets the [file metadata](members.md#m_metadata) to metadata.

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QVariant]: http://doc.qt.io/qt-5/qvariant.html "QVariant"
[QVariantMap]: http://doc.qt.io/qt-5/qvariant.html#QVariantMap-typedef "QVariantMap"
