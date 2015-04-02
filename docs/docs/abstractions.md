# C++ Plugin API

The in-depth documentation for the classes and abstractions in the C++ plugin API.

---

## File

A file path with associated metadata.

The File is one of two important data structures in OpenBR (the Template is the other).
It is typically used to store the path to a file on disk with associated metadata.
The ability to associate a key/value metadata table with the file helps keep the API simple while providing customizable behavior.

When querying the value of a metadata key, the value will first try to be resolved against the file's private metadata table.
If the key does not exist in its local table then it will be resolved against the properties in the global Context.
By design file metadata may be set globally using Context::setProperty to operate on all files.

Files have a simple grammar that allow them to be converted to and from strings.
If a string ends with a \c ] or \c ) then the text within the final \c [] or \c () are parsed as comma separated metadata fields.
By convention, fields within \c [] are expected to have the format <tt>[key1=value1, key2=value2, ..., keyN=valueN]</tt> where order is irrelevant.
Fields within \c () are expected to have the format <tt>(value1, value2, ..., valueN)</tt> where order matters and the key context dependent.
The left hand side of the string not parsed in a manner described above is assigned to #name.

Values are not necessarily stored as strings in the metadata table.
The system will attempt to infer and convert them to their "native" type.
The conversion logic is as follows:

1. If the value starts with \c [ and ends with \c ] then it is treated as a comma separated list and represented with \c QVariantList. Each value in the list is parsed recursively.
2. If the value starts with \c ( and ends with \c ) and contains four comma separated elements, each convertable to a floating point number, then it is represented with \c QRectF.
3. If the value starts with \c ( and ends with \c ) and contains two comma separated elements, each convertable to a floating point number, then it is represented with \c QPointF.
4. If the value is convertable to a floating point number then it is represented with \c float.
5. Otherwise, it is represented with \c QString.

Metadata keys fall into one of two categories:
* \c camelCaseKeys are inputs that specify how to process the file.
* \c Capitalized_Underscored_Keys are outputs computed from processing the file.

Below are some of the most commonly occuring standardized keys:

Key             | Value          | Description
---             | ----           | -----------
name            | QString        | Contents of #name
separator       | QString        | Seperate #name into multiple files
Index           | int            | Index of a template in a template list
Confidence      | float          | Classification/Regression quality
FTE             | bool           | Failure to enroll
FTO             | bool           | Failure to open
*_X             | float          | Position
*_Y             | float          | Position
*_Width         | float          | Size
*_Height        | float          | Size
*_Radius        | float          | Size
Label           | QString        | Class label
Theta           | float          | Pose
Roll            | float          | Pose
Pitch           | float          | Pose
Yaw             | float          | Pose
Points          | QList<QPointF> | List of unnamed points
Rects           | QList<Rect>    | List of unnamed rects
Age             | float          | Age used for demographic filtering
Gender          | QString        | Subject gender
Train           | bool           | The data is for training, as opposed to enrollment
_*              | *              | Reserved for internal use

### Members

#### Name

Path to a file on disk

#### FTE

'Failed to enroll'. If true this file failed to be processed somewhere in the template enrollment algorithm

#### m_metadata

Map for storing metadata. It is a <a href=http://doc.qt.io/qt-5/QString.html>QString</a>, <a href=http://doc.qt.io/qt-5/qvariant.html>QVariant</a> key value pairing.

---

## Template

---

## Transform

---

## UntrainableTransform

---

## MetaTransform

---

## UntrainableMetaTransform

---

## MetadataTransform

---

## UntrainableMetadataTransform

---

## TimeVaryingTransform

---

## Distance

---

## UntrainableDistance

---

## Output

---

## MatrixOutput

---

## Format

---

## Representation

---

## Classifier
