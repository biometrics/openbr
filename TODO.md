# Removed Core functionality

- boost.cpp This will likely be the most challenging port. OpenCV 4 carries the old ml headers in the traincascade application (presumably so they don't have to do this port)

# Removed plugins

- classification/boostedforest.cpp: Relies on core boost functionality
- classification/forest.cpp: Big changes to the RTree interface. It's unclear that ForestInduction is possible in the new interface at all. If it is, it would likely require walking the tree node lists manually
- imgproc/custom_sift.cpp: Uses functions `fastAtan2`, `magnitude`, and `exp` which are OpenCV functions but with a different set of args (float arrays vs. floats). I can't find a record of float* functions in OpenCV 2 either
                           so it's unclear where those are defined.
- imgproc/keypointdescriptor.cpp: OpenCV 4 totally changes the interface from a `create(string)` model to mirror the ml `Ptr<DescriptionExtractor> p = Subclass::create()` paradigm. While this would be relatively easy to
                                  implement as a enum/switch statement, most of the subclasses have custom initialization parameters. Unclear what the best way to handle those is
- metadata/keypointdetector.cpp: Same as above 
