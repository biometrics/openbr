# Algorithms in OpenBR

**So you've run `scripts/helloWorld.sh` and it generally makes sense, except you have no idea what    `'Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+CvtFloat+PCA(0.95):Dist(L2)'` means or how it is executed.**

Well if this is the case, you've found the right documentation.
Let's get started!

In OpenBR an *algorithm* is a technique for enrolling templates associated with a technique for comparing them.
Recall that our ultimate goal is to be able to say how similar two face images are (or two fingerprints, irises, etc.).
Instead of storing the entire raw image for comparison, it is common practice to store an optimized representation, or *template*, of the image for the task at hand.
The process of generating this optimized representation is called *template enrollment* or *template generation*.
Given two templates, *template comparison* computes the similarity between them, where the higher values indicate more probable matches and the threshold for determining what constitutes an adequate match is determined operationally.
The goal of template generation is to design templates that are small, accurate, and fast to compare.
Ok, you probably knew all of this already, let's move on.

The only way of creating an algorithm in OpenBR is from a text string that describes it.
We call this string the *algorithm description*.
The algorithm description is separated into two parts by a ':', with the left hand side indicating how to generate templates and the right hand side indicating how to compare them.
Some algorithms, like [age_estimation](tutorials.md#age estimation) and [gender estimation](tutorials.md#gender estimation) are *classifiers* that don't create templates.
In these cases, the colon and the template comparison technique can be omitted from the algorithm description.

There are several motivations for mandating that algorithms are defined from these strings, here are the most important:
    1. It ensures good software development practices by forcibly decoupling the development of each step in an algorithm, facilitating the modification of algorithms and the re-use of individual steps.
    2. It spares the creation and maintenance of a lot of very similar header files that would otherwise be needed for each step in an algorithm, observe the absence of headers in `openbr/plugins`.
    3. It allows for algorithm parameter tuning without recompiling.
    4. It is completely unambiguous, both the OpenBR interpreter and anyone familiar with the project can understand exactly what your algorithm does just from this description.

Let's look at some of the important parts of the code base that make this possible!

In `AlgorithmCore::init()` in `openbr/core/core.cpp` you can see the code for splitting the algorithm description at the colon.
Shortly thereafter in this function we *make* the template generation and comparison methods.
These make calls are defined in the public [C++ plugin API](#the c++ plugin api) and can also be called from end user code.

Below we discuss some of the source code for `Transform::make` in `openbr/openbr_plugin.cpp`.
Note, the make functions for other plugin types are similar in spirit and will not be covered.

One of the first steps when converting the template enrollment description into a [Transform](docs/cpp_api.md#Transform) is to replace the operators, like '+', with their full form:

    { // Check for use of '+' as shorthand for Pipe(...)
         QStringList words = parse(str, '+');
         if (words.size() > 1)
             return make("Pipe([" + words.join(",") + "])", parent);
    }

A pipe (see [PipeTransform](docs/plugins/core.md#pipetransform)) is the standard way of chaining together multiple steps in series to form more sophisticated algorithms.
PipeTransform takes a list of transforms, and *projects* templates through each transform in order.

After operator expansion, the template enrollment description forms a tree, and the transform is constructed from this description starting recursively starting at the root of the tree:

    Transform *transform = Factory<Transform>::make("." + str);

At this point we reach arguably the most important code in the entire framework, the *object factory* in `openbr/openbr_plugin.h`.
The [Factory](docs/cpp_api.md#factory) class is responsible for constructing an object from a string:

    static T *make(const File &file)
    {
        QString name = file.get<QString>("plugin", "");
        if (name.isEmpty()) name = file.suffix();
        if (!names().contains(name)) {
            if      (names().contains("Empty") && name.isEmpty()) name = "Empty";
            else if (names().contains("Default"))                 name = "Default";
            else    qFatal("%s registry does not contain object named: %s", qPrintable(baseClassName()), qPrintable(name));
        }
        T *object = registry->value(name)->_make();
        static_cast<Object*>(object)->init(file);
        return object;
    }

Going back to our original example, a [PipeTransform](docs/plugins/core.md#pipetransform) will be created with [OpenTransform](docs/plugins/io.md#opentransform), [CvtTransform](docs/plugins/imgproc.md#cvttransform), [CascadeTransform](docs/plugins/metadata.md#cascadetransform), [ASEFEyesTransform](docs/plugins/metadata.md#asefeyestransform), [AffineTransform](docs/plugins/imgproc.md#affinetransform), [CvtFloatTransform](docs/plugins/imgproc.md#cvtfloattransform), and [PCATransform](docs/plugins/classification.md#pcatransform) as its children.

If you want all the tedious details about what exactly this algoritm does, then you should read the [project](docs/cpp_api.md#project) function implemented by each of these plugins.
The brief explanation is that it *reads the image from disk, converts it to grayscale, runs the face detector, runs the eye detector on any detected faces, uses the eye locations to normalize the face for rotation and scale, converts to floating point format, and then embeds it in a PCA subspaced trained on face images*.
If you are familiar with face recognition, you will likely recognize this as the Eigenfaces \cite turk1991eigenfaces algorithm.

As a final note, the Eigenfaces algorithms uses the Euclidean distance (or L2-norm) to compare templates.
Since OpenBR expects *similarity* values when comparing templates, and not *distances*, the [DistDistance](docs/plugins/distance.md#distdistance) will return *-log(distance+1)* so that larger values indicate more similarity.
