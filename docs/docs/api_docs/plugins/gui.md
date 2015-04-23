# AdjacentOverlayTransform

Load the image named in the specified property, draw it on the current matrix adjacent to the rect specified in the other property.
 

* **file:** gui/adjacentoverlay.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# DrawDelaunayTransform

Creates a Delaunay triangulation based on a set of points
 

* **file:** gui/drawdelaunay.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# DrawGridLinesTransform

Draws a grid on the image
 

* **file:** gui/drawgridlines.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DrawOpticalFlow

Draw a line representing the direction and magnitude of optical flow at the specified points.
 

* **file:** gui/drawopticalflow.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# DrawPropertiesPointTransform

Draw the values of a list of properties at the specified point on the image

The inPlace argument controls whether or not the image is cloned before it is drawn on.
 

* **file:** gui/drawpropertiespoint.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# DrawPropertyPointTransform

Draw the value of the specified property at the specified point on the image

The inPlace argument controls whether or not the image is cloned before it is drawn on.
 

* **file:** gui/drawpropertypoint.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# DrawSegmentation

Fill in the segmentations or draw a line between intersecting segments.
 

* **file:** gui/drawsegmentation.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# DrawTransform

Renders metadata onto the image.

The inPlace argument controls whether or not the image is cloned before the metadata is drawn.
 

* **file:** gui/draw.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ElicitTransform

Elicits metadata for templates in a pretty GUI
 

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](#showtransform)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# FPSCalc

Calculates the average FPS of projects going through this transform, stores the result in AvgFPS
Reports an average FPS from the initialization of this transform onwards.
 

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# FPSLimit

Limits the frequency of projects going through this transform to the input targetFPS
 

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# FilterTransform

DOCUMENT ME
 

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](#showtransform)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# ManualRectsTransform

Manual select rectangular regions on an image.
Stores marked rectangles as anonymous rectangles, or if a set of labels is provided, prompt the user
to select one of those labels after drawing each rectangle.
 

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](#showtransform)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# ManualTransform

Manual selection of landmark locations
 

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](#showtransform)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# ShowTrainingTransform

Show the training data
 

* **file:** gui/show.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ShowTransform

Displays templates in a GUI pop-up window using QT.

Can be used with parallelism enabled, although it is considered TimeVarying.
 

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# SurveyTransform

Display an image, and asks a yes/no question about it
 

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](#showtransform)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

