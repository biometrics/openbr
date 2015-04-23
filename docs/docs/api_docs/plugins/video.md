# AggregateFrames

Passes along n sequential frames to the next [Transform](../cpp_api/transform/transform.md).

For a video with m frames, AggregateFrames would create a total of m-n+1 sequences ([0,n] ... [m-n+1, m])
 

* **file:** video/aggregate.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DropFrames

Only use one frame every n frames.
 

* **file:** video/drop.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Austin Blanton][imaus10
  
   For a video with m frames, DropFrames will pass on m/n frames.]
* **properties:** None


---

# OpticalFlowTransform

Gets a one-channel dense optical flow from two images
 

* **file:** video/opticalflow.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

