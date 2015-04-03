---

# AdjacentOverlayTransform

Load the image named in the specified property, draw it on the current matrix adjacent to the rect specified in the other property.

* **file:** gui/adjacentoverlay.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Charles Otto
* **properties:** None

---

# DrawTransform

Renders metadata onto the image.

* **file:** gui/draw.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# DrawDelaunayTransform

Creates a Delaunay triangulation based on a set of points

* **file:** gui/drawdelaunay.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

# DrawGridLinesTransform

Draws a grid on the image

* **file:** gui/drawgridlines.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# DrawOpticalFlow

Draw a line representing the direction and magnitude of optical flow at the specified points.

* **file:** gui/drawopticalflow.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

# DrawPropertiesPointTransform

Draw the values of a list of properties at the specified point on the image

* **file:** gui/drawpropertiespoint.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Charles Otto
* **properties:** None

---

# DrawPropertyPointTransform

Draw the value of the specified property at the specified point on the image

* **file:** gui/drawpropertypoint.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Charles Otto
* **properties:** None

---

# DrawSegmentation

Fill in the segmentations or draw a line between intersecting segments.

* **file:** gui/drawsegmentation.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

# ShowTransform

Displays templates in a GUI pop-up window using QT.

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

---

# ShowTrainingTransform

Show the training data

* **file:** gui/show.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

# ManualTransform

Manual selection of landmark locations

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](../cpp_api.md#showtransform)
* **author:** Scott Klum
* **properties:** None

---

# ManualRectsTransform

Manual select rectangular regions on an image.

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](../cpp_api.md#showtransform)
* **author:** Charles Otto
* **properties:** None

---

# ElicitTransform

Elicits metadata for templates in a pretty GUI

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](../cpp_api.md#showtransform)
* **author:** Scott Klum
* **properties:** None

---

# SurveyTransform

Display an image, and asks a yes/no question about it

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](../cpp_api.md#showtransform)
* **author:** Charles Otto
* **properties:** None

---

# FPSLimit

Limits the frequency of projects going through this transform to the input targetFPS

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

---

# FPSCalc

Calculates the average FPS of projects going through this transform, stores the result in AvgFPS

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

