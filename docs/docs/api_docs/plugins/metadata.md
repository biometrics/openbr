# ASEFEyesTransform

Find eye locations using an ASEF filter
 

* **file:** metadata/eyes.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Bolme, D.S.; Draper, B.A.; Beveridge, J.R.;*
	 **"Average of Synthetic Exact Filters,"**
	 Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on , vol., no., pp.2105-2112, 20-25 June 2009

* **properties:** None


---

# AnonymizeLandmarksTransform

Remove a name from a point/rect
 

* **file:** metadata/anonymizelandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# AsTransform

Change the [Template](../cpp_api/template/template.md) file extension
 

* **file:** metadata/as.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# AveragePointsTransform

Averages a set of landmarks into a new landmark
 

* **file:** metadata/averagepoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# CascadeTransform

Wraps OpenCV cascade classifier
 

* **file:** metadata/cascade.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Josh Klontz][jklontz], [David Crouse][dgcrouse]
* **see:** [http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html](http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html)
* **properties:** None


---

# CheckTransform

Checks the [Template](../cpp_api/template/template.md) for NaN values.
 

* **file:** metadata/check.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ClearPointsTransform

Clears the points from a [Template](../cpp_api/template/template.md)
 

* **file:** metadata/clearpoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# ConsolidateDetectionsTransform

Consolidate redundant/overlapping detections.
 

* **file:** metadata/consolidatedetections.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# CropRectTransform

Crops the width and height of a [Template](../cpp_api/template/template.md) rects by input width and height factors.
 

* **file:** metadata/croprect.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# DelaunayTransform

Creates a Delaunay triangulation based on a set of points
 

* **file:** metadata/delaunay.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# ExpandRectTransform

Expand the width and height of a [Template](../cpp_api/template/template.md) rects by input width and height factors.
 

* **file:** metadata/expandrect.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# ExtractMetadataTransform

Create matrix from metadata values.
 

* **file:** metadata/extractmetadata.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FaceFromEyesTransform

Create face bounding box from two eye locations.
 

* **file:** metadata/facefromeyes.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	widthPadding | double | Specifies what percentage of the interpupliary distance (ipd) will be padded in both horizontal directions.
	verticalLocation | double | specifies where vertically the eyes are within the bounding box (0.5 would be the center).

---

# FileExclusionTransform

DOCUMENT ME
 

* **file:** metadata/fileexclusion.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Unknown][Unknown]
* **properties:** None


---

# FilterDupeMetadataTransform

Removes duplicate [Template](../cpp_api/template/template.md) based on a unique metadata key
 

* **file:** metadata/filterdupemetadata.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# GridTransform

Add landmarks to the [Template](../cpp_api/template/template.md) in a grid layout
 

* **file:** metadata/grid.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GroundTruthTransform

Add any ground truth to the [Template](../cpp_api/template/template.md) using the file's base name.
 

* **file:** metadata/groundtruth.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# HOGPersonDetectorTransform

Detects objects with OpenCV's built-in HOG detection.
 

* **file:** metadata/hogpersondetector.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **see:** [http://docs.opencv.org/modules/gpu/doc/object_detection.html](http://docs.opencv.org/modules/gpu/doc/object_detection.html)
* **properties:** None


---

# IfMetadataTransform

Clear [Template](../cpp_api/template/template.md) without the required metadata.
 

* **file:** metadata/ifmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ImpostorUniquenessMeasureTransform

Impostor Uniqueness Measure
 

* **file:** metadata/imposteruniquenessmeasure.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# JSONTransform

Represent the metadata as JSON template data.
 

* **file:** metadata/json.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# KeepMetadataTransform

Retains only the values for the keys listed, to reduce template size
 

* **file:** metadata/keepmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# KeyPointDetectorTransform

Wraps OpenCV Key Point Detector
 

* **file:** metadata/keypointdetector.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html](http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html)
* **properties:** None


---

# KeyToRectTransform

Convert values of key_X, key_Y, key_Width, key_Height to a rect.
 

* **file:** metadata/keytorect.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Jordan Cheney][JordanCheney]
* **properties:** None


---

# NameLandmarksTransform

Name a point/rect
 

* **file:** metadata/namelandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# NameTransform

Sets the [Template](../cpp_api/template/template.md) matrix data to the br::File::name.
 

* **file:** metadata/name.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# NormalizePointsTransform

Normalize points to be relative to a single point
 

* **file:** metadata/normalizepoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# PointDisplacementTransform

Normalize points to be relative to a single point
 

* **file:** metadata/pointdisplacement.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# PointsToMatrixTransform

Converts either the file::points() list or a QList<QPointF> metadata item to be the template's matrix
 

* **file:** metadata/pointstomatrix.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# ProcrustesTransform

Procrustes alignment of points
 

* **file:** metadata/procrustes.cpp
* **inherits:** [MetadataTransform](../cpp_api/metadatatransform/metadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# RectsToTemplatesTransform

For each rectangle bounding box in src, a new [Template](../cpp_api/template/template.md) is created.
 

* **file:** metadata/rectstotemplates.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# RegexPropertyTransform

Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.
 

* **file:** metadata/regexproperty.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# RemoveMetadataTransform

Removes a metadata field from all [Template](../cpp_api/template/template.md)
 

* **file:** metadata/removemetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# RemoveTemplatesTransform

Remove [Template](../cpp_api/template/template.md) with the specified file extension or metadata value.
 

* **file:** metadata/removetemplates.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RenameFirstTransform

Rename first found metadata key
 

* **file:** metadata/renamefirst.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RenameTransform

Rename metadata key
 

* **file:** metadata/rename.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ReorderPointsTransform

Reorder the points such that points[from[i]] becomes points[to[i]] and
vice versa
 

* **file:** metadata/reorderpoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# RestoreMatTransform

Set the last matrix of the input [Template](../cpp_api/template/template.md) to a matrix stored as metadata with input propName.

Also removes the property from the [Template](../cpp_api/template/template.md)s metadata after restoring it.
 

* **file:** metadata/restoremat.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# SaveMatTransform

Store the last matrix of the input [Template](../cpp_api/template/template.md) as a metadata key with input property name.
 

* **file:** metadata/savemat.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# SelectPointsTransform

Retains only landmarks/points at the provided indices
 

* **file:** metadata/selectpoints.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# SetMetadataTransform

Sets the metadata key/value pair.
 

* **file:** metadata/setmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# SetPointsInRectTransform

Set points relative to a rect
 

* **file:** metadata/setpointsinrect.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Jordan Cheney][JordanCheney]
* **properties:** None


---

# StasmTransform

Wraps STASM key point detector
 

* **file:** metadata/stasm4.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# StopWatchTransform

Gives time elapsed over a specified [Transform](../cpp_api/transform/transform.md) as a function of both images (or frames) and pixels.
 

* **file:** metadata/stopwatch.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Jordan Cheney][JordanCheney], [Josh Klontz][jklontz]
* **properties:** None


---

