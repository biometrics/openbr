---

# AnonymizeLandmarksTransform

Remove a name from a point/rect

* **file:** metadata/anonymizelandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# AsTransform

Change the br::Template::file extension

* **file:** metadata/as.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

# AveragePointsTransform

Averages a set of landmarks into a new landmark

* **file:** metadata/averagepoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

# CascadeTransform

Wraps OpenCV cascade classifier

* **file:** metadata/cascade.cpp
* **inherits:** [MetaTransform](../cpp_api.md#metatransform)
* **authors:** Josh Klontz, David Crouse
* **properties:** None

---

# CheckTransform

Checks the template for NaN values.

* **file:** metadata/check.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

# ClearPointsTransform

Clears the points from a template

* **file:** metadata/clearpoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

# ConsolidateDetectionsTransform

Consolidate redundant/overlapping detections.

* **file:** metadata/consolidatedetections.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

# CropRectTransform

Crops the width and height of a template's rects by input width and height factors.

* **file:** metadata/croprect.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# DelaunayTransform

Creates a Delaunay triangulation based on a set of points

* **file:** metadata/delaunay.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

# ExpandRectTransform

Expand the width and height of a template's rects by input width and height factors.

* **file:** metadata/expandrect.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Charles Otto
* **properties:** None

---

# ExtractMetadataTransform

Create matrix from metadata values.

* **file:** metadata/extractmetadata.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

# ASEFEyesTransform

Bolme, D.S.; Draper, B.A.; Beveridge, J.R.;

* **file:** metadata/eyes.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **authors:** David Bolme, Josh Klontz
* **properties:** None

---

# FilterDupeMetadataTransform

Removes duplicate templates based on a unique metadata key

* **file:** metadata/filterdupemetadata.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api.md#timevaryingtransform)
* **author:** Austin Blanton
* **properties:** None

---

# GridTransform

Add landmarks to the template in a grid layout

* **file:** metadata/grid.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# GroundTruthTransform

Add any ground truth to the template using the file's base name.

* **file:** metadata/groundtruth.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

# HOGPersonDetectorTransform

Detects objects with OpenCV's built-in HOG detection.

* **file:** metadata/hogpersondetector.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

# IfMetadataTransform

Clear templates without the required metadata.

* **file:** metadata/ifmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

# ImpostorUniquenessMeasureTransform

Impostor Uniqueness Measure

* **file:** metadata/imposteruniquenessmeasure.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

# JSONTransform

Represent the metadata as JSON template data.

* **file:** metadata/json.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

# KeepMetadataTransform

Retains only the values for the keys listed, to reduce template size

* **file:** metadata/keepmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# KeyPointDetectorTransform

Wraps OpenCV Key Point Detector

* **file:** metadata/keypointdetector.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# KeyToRectTransform

Convert values of key_X, key_Y, key_Width, key_Height to a rect.

* **file:** metadata/keytorect.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Jordan Cheney
* **properties:** None

---

# MongooseInitializer

Initialize mongoose server

* **file:** metadata/mongoose.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **author:** Unknown
* **properties:** None

---

# NameTransform

Sets the template's matrix data to the br::File::name.

* **file:** metadata/name.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

# NameLandmarksTransform

Name a point/rect

* **file:** metadata/namelandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# NormalizePointsTransform

Normalize points to be relative to a single point

* **file:** metadata/normalizepoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# PointDisplacementTransform

Normalize points to be relative to a single point

* **file:** metadata/pointdisplacement.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# PointsToMatrixTransform

Converts either the file::points() list or a QList<QPointF> metadata item to be the template's matrix

* **file:** metadata/pointstomatrix.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

# ProcrustesTransform

Procrustes alignment of points

* **file:** metadata/procrustes.cpp
* **inherits:** [MetadataTransform](../cpp_api.md#metadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# RectsToTemplatesTransform

For each rectangle bounding box in src, a new

* **file:** metadata/rectstotemplates.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Brendan Klare
* **properties:** None

---

# RegexPropertyTransform

Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.

* **file:** metadata/regexproperty.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Charles Otto
* **properties:** None

---

# RemoveMetadataTransform

Removes a metadata field from all templates

* **file:** metadata/removemetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

# RemoveTemplatesTransform

Remove templates with the specified file extension or metadata value.

* **file:** metadata/removetemplates.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

# RenameTransform

Rename metadata key

* **file:** metadata/rename.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

# RenameFirstTransform

Rename first found metadata key

* **file:** metadata/renamefirst.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

# ReorderPointsTransform

Reorder the points such that points[from[i]] becomes points[to[i]] and

* **file:** metadata/reorderpoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

# RestoreMatTransform

Set the last matrix of the input template to a matrix stored as metadata with input propName.

* **file:** metadata/restoremat.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:** None

---

# SaveMatTransform

Store the last matrix of the input template as a metadata key with input property name.

* **file:** metadata/savemat.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:** None

---

# SelectPointsTransform

Retains only landmarks/points at the provided indices

* **file:** metadata/selectpoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

# SetMetadataTransform

Sets the metadata key/value pair.

* **file:** metadata/setmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

# SetPointsInRectTransform

Set points relative to a rect

* **file:** metadata/setpointsinrect.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api.md#untrainablemetadatatransform)
* **author:** Jordan Cheney
* **properties:** None

---

# StasmInitializer

Initialize Stasm

* **file:** metadata/stasm4.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **author:** Scott Klum
* **properties:** None

---

# StasmTransform

Wraps STASM key point detector

* **file:** metadata/stasm4.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

# StopWatchTransform

Gives time elapsed over a specified transform as a function of both images (or frames) and pixels.

* **file:** metadata/stopwatch.cpp
* **inherits:** [MetaTransform](../cpp_api.md#metatransform)
* **authors:** Jordan Cheney, Josh Klontz
* **properties:** None

