
Welcome to OpenBR! Here we have a series of tutorials designed to get you up to speed on what OpenBR is, how it works, its command line interface, and the C API. These tutorials aren't meant to be completed in a specific order so feel free to hop around. If you need help, feel free to [contact us](index.md#help).

---

# Quick Start

This tutorial is meant to familiarize you with the ideas, objects and motivations behind OpenBR using some fun examples. **Note:** parts of this tutorial require a webcam.

OpenBR is a C++ library built on top of QT and OpenCV. It can either be used from the command line using the **br** application, or from interfacing with the [C API](api_docs/c_api.md). The command line is the easiest and fastest way to get started and this tutorial will use it for all of the examples.

First, make sure that OpenBR has been installed on your system using the steps described in the [installation section](install.md). Then open up your terminal or command prompt and enter:

    $ br -gui -algorithm "Show(false)" -enroll 0.webcam

If everything goes well your webcam should have opened up and is streaming. Look you are using OpenBR! Let's talk about what's happening in this command. OpenBR expects flags to be prepended by a *-* and arguments to follow the flags and be separated by spaces. Flags normally require a specific number of flags. All of the possible flags and their values are [documented here](api_docs/cl_api.md). Let's step through the individual arguments and values. **-gui** is the flag that tells OpenBR to open up a GUI window. Take a look at the [GUI plugins](api_docs/plugins/gui.md) for other plugins that require the **-gui** flag. **-algorithm** is one of the most important flags in OpenBR. It expects one argument called the *algorithm string*. This string determines the pipeline that images and metadata propagate through. Finally, **-enroll** reads files from disk and *enrolls* them into the image pipeline. It takes one input argument (0.webcam in our example) and an optional output argument. OpenBR has a range of formats that can be enrolled into algorithms, some examples include .jpg, .png, .csv, and .xml. .webcam tells OpenBR to enroll frames from the computers webcam.

Let's try a slightly more complicated example, after all OpenBR can do way more then just open webcams! Fire up the terminal again and enter:

    $ br -gui -algorithm "Cvt(Gray)+Show(false)" -enroll 0.webcam

Hey what happened? We took our normal BGR (OpenCV's alternative to RGB) image and converted it to a grayscale image. How did we do that? Simple, by adding "Cvt(Gray)" to the algorithm string. [Cvt](api_docs/plugins/imgproc.md#cvttransform), short for convert, is an example of an OpenBR *plugin*. [Show](api_docs/plugins/gui.md#showtransform) is a plugin as well. Every algorithm string in OpenBR is just a series of plugins joined together into a pipeline. In fact the **+** symbol is shorthand for a [Pipe](api_docs/plugins/core.md#pipetransform), another kind of OpenBR plugin. We specify **Gray** to [Cvt](api_docs/plugins/imgproc.md#cvttransform) as a runtime parameter to tell the plugin which color space to convert the image to. We also could have written **Cvt(HSV)** if we wanted to convert to the HSV color space or **Cvt(Luv)** if we wanted to convert to Luv. The arguments inside of the plugins are runtime parameters that can adjust the functionality. They can be provided as key-value pairs, **Cvt(Gray)** is equivalent to **Cvt(ColorSpace=Gray)**, or as keyless values. Make sure you are supplying parameters in the proper order if you are not using keys! Try and run the code with **Show(true)** and see how changing the parameters effect the output of the command (**Hint:** hit a key to cycle through the images).

Let's make this example a little bit more exciting and relevant to OpenBR's biometric roots. Face detection is normally the first step in a [face recognition](#face-recognition) algorithm. Let's do face detection in OpenBR. Back in the terminal enter:

    $ br -gui -algorithm "Cvt(Gray)+Cascade(FrontalFace)+Draw(lineThickness=3)+Show(false)" -enroll 0.webcam

You're webcam should be open again but this time a box should have appeared around your face! We added two new plugins to our string, [Cascade](api_docs/plugins/metadata.md#cascadetransform) and [Draw](api_docs/plugins/gui.md#drawtransform). Let's walk through this plugin by plugin and see how it works:

1. [Cvt(Gray)](api_docs/plugins/imgproc.md#cvttransform): Convert the image from BGR to grayscale. Grayscale is required for [Cascade](api_docs/plugins/metadata.md#cascadetransform) to work properly.
2. [Cascade(FrontalFace)](api_docs/plugins/metadata.md#cascadetransform): This is a wrapper on the OpenCV [Cascade Classification](http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html) framework. It detects frontal faces using the **FrontalFace** model.
3. [Draw(lineThickness=3)](api_docs/plugins/gui.md#drawtransform): Take the rects detected by [Cascade](api_docs/plugins/metadata.md#cascadetransform) and draw them onto the frame from the webcam. **lineThickness** determines the thickness of the drawn rect.
4. [Show(false)](api_docs/plugins/gui.md#showtransform): Show the image in a GUI window. **false** indicates the images should be shown in succession without waiting for a key press.

Pretty straightforward right? Each plugin completes one task and the passes the output on to the next plugin. You can pipe together as many plugins as you like as long as the output data from one can be the input data to the next. But wait! Output data? Input data? we haven't talked about data at all yet! How does OpenBR handle data? There are two objects that handle data is OpenBR; [Files](api_docs/cpp_api/file/file.md), which handle metadata, and [Templates](api_docs/cpp_api/template/template.md) which are containers for images and [Files](api_docs/cpp_api/file/file.md). Let's talk about [Files](api_docs/cpp_api/file/file.md) first. A file consists of file name, which is a path to a file on disk, and metadata which is a map of key-value pairs. The metadata can contain any textual information about the file. In the example above we use it to store the rectangles detected by [Cascade](api_docs/plugins/metadata.md#cascadetransform) and pass them along to [Draw](api_docs/plugins/gui.md#drawtransform) for drawing. [Templates](api_docs/cpp_api/template/template.md) are containers for images, given as OpenCV [Mats][Mat] and [Files](api_docs/cpp_api/file/file.md). They can contain one image or a list of images. Plugins are either [Template](api_docs/cpp_api/template/template.md) in, [Template](api_docs/cpp_api/template/template.md) out or [TemplateList](api_docs/cpp_api/templatelist/templatelist.md) in, [TemplateList](api_docs/cpp_api/templatelist/templatelist.md) out. [TemplateLists](api_docs/cpp_api/templatelist/templatelist.md) are, of course, just a list of [Templates](api_docs/cpp_api/template/template.md) which a few functions added for your convenience.

And there you go! You have gotten your quick start in OpenBR. We covered the [command line](api_docs/cl_api.md), [plugins, and the key data structures](api_docs/cpp_api.md) in OpenBR. If you want to learn more the next few tutorials will cover these fields with more depth.

---

# Algorithms in OpenBR

The purpose of OpenBR is to express biometrics algorithms in a consistent and simple way. But what exactly does *algorithm* mean? In OpenBR an *algorithm* is a technique for enrolling templates associated with a technique for comparing them. Recall that our ultimate goal is to be able to say how similar two face images are (or two fingerprints, irises, etc.).

Instead of storing the entire raw image for comparison, it is common practice to store an optimized representation, or *template*, of the image for the task at hand. The process of generating this optimized representation is called *template enrollment* or *template generation*. Given two templates, *template comparison* computes the similarity between them, where the higher values indicate more probable matches and the threshold for determining what constitutes an adequate match is determined operationally. The goal of template generation is to design templates that are small, accurate, and fast to compare.

The only way of creating an algorithm in OpenBR is from a text string that describes it. We call this string the *algorithm description*. There are several motivations for mandating that algorithms are defined from these strings, here are the most important:

1. It ensures good software development practices by forcibly decoupling the development of each step in an algorithm, facilitating the modification of algorithms and the re-use of individual steps.
2. It spares the creation and maintenance of a lot of very similar header files that would otherwise be needed for each step in an algorithm, observe the absence of headers in `openbr/plugins`.
3. It allows for algorithm parameter tuning without recompiling.
4. It is completely unambiguous, both the OpenBR interpreter and anyone familiar with the project can understand exactly what your algorithm does just from this description.

OpenBR provides a syntax for creating more concise algorithm descriptions. The relevant symbols are:

Symbol | Meaning
--- | ---
PluginName(property1=value1,...propertyN=valueN) | A plugin is described by its name (without the abstraction) and a list of properties and values. Properties of a plugin that are not specified are set to their default values.
: | Seperates *templatelate enrollment* from *template comparison*. Enrollment is on the left and comparison is on the right. This is optional, algorithms without comparisons are called *classifiers*
\+ | Pipe. Joins plugins together and projects input through them in series. The output of a plugin to the left of \+ become the input of the plugin to the right.
 / | Fork. Joins plugins together and projects input through them in parallel. All plugins get the same input and the output is concattenated together.
 \{\} | Cache. Cache the output of a plugin in memory for quick lookups later on
 <\> | LoadStore. Plugins inside are stored on disk after training and / or loaded from disk before projection
 () | Order of operations. Change the order of operations using parantheses.

Let's look at some of the important parts of the code base that make this possible!

In ```AlgorithmCore::init()``` in ```openbr/core/core.cpp``` you can see the code for splitting the algorithm description at the colon.
Shortly thereafter in this function we *make* the template generation and comparison methods.
These make calls are defined in the public [C++ plugin API](api_docs/cpp_api.md) and can also be called from end user code.

Below we discuss some of the source code for `Transform::make` in `openbr/openbr_plugin.cpp`.
Note, the make functions for other plugin types are similar in spirit and will not be covered.

One of the first steps when converting the template enrollment description into a [Transform](api_docs/cpp_api/transform/transform.md) is to replace the operators, like '\+', with their full form:

    { // Check for use of '+' as shorthand for Pipe(...)
         QStringList words = parse(str, '+');
         if (words.size() > 1)
             return make("Pipe([" + words.join(",") + "])", parent);
    }

After operator expansion, the template enrollment description forms a tree, and the transform is constructed from this description starting recursively starting at the root of the tree:

    Transform *transform = Factory<Transform>::make("." + str);

Let's use the algorithm in ```scripts/helloWorld.sh``` as an example. The algorithm is:

    Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+CvtFloat+PCA(0.95):Dist(L2)

So what is happening here? Let's expand this using our new knowledge of OpenBR's algorithm syntax. First, the algorithm will be split into enrollment and comparison portions at the **:**. So enrollment becomes-

    Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+CvtFloat+PCA(0.95)

and comparison is-

    Dist(L2)

On the enrollment side the **+'s** are converted into a [PipeTransform](api_docs/plugins/core.md#pipetransform) with the other plugins as children. Enrollment is transformed to

    Pipe(transforms=[Open,Cvt(Gray),Cascade(FrontalFace),ASEFEyes,Affine(128,128,0.33,0.45,CvtFloat,PCA(0.95)])

If you want all the tedious details about what exactly this algoritm does, then you should read the [project](api_docs/cpp_api/transform/functions.md#project-1) function implemented by each of these plugins.
The brief explanation is that it *reads the image from disk, converts it to grayscale, runs the face detector, runs the eye detector on any detected faces, uses the eye locations to normalize the face for rotation and scale, converts to floating point format, and then embeds it in a PCA subspaced trained on face images*.
If you are familiar with face recognition, you will likely recognize this as the Eigenfaces[^1] algorithm.

As a final note, the Eigenfaces algorithms uses the Euclidean distance (or L2-norm) to compare templates.
Since OpenBR expects *similarity* values when comparing templates, and not *distances*, the [DistDistance](api_docs/plugins/distance.md#distdistance) will return *-log(distance+1)* so that larger values indicate more similarity.

---

# Training Algorithms

---

# The Evaluation Harness

OpenBR implements a complete, [NIST](http://www.nist.gov/index.html) compliant, evaluation harness for evaluating face recognition, face detection, and facial landmarking. The goal is to provide a consistent environment for the repeatable evaluation of algorithms to the academic and open source communities. To accompish this OpenBR defines the following portions of the biometrics evaluation evironment (BEE) standard-

* Signature Set- A signature set (or *sigset*) is a [Gallery](api_docs/cpp_api/gallery/gallery.md) compliant **XML** file-list specified on page 9 of the [MBGC File Overview](misc/MBGC_file_overview.pdf) and implemented in [xmlGallery](api_docs/plugins/gallery.md#xmlgallery). Sigsets are identified with a **.xml** extension.

* Similarity Matrix- A similarity matrix (or *simmat*) is an [Output](api_docs/cpp_api/output/output.md) compliant binary score matrix specified on page 12 of the [MBGC File Overview](misc/MBGC_file_overview.pdf) and implemented in [mtxOutput](api_docs/plugins/output.md#mtxoutput). Simmats are identified with a **.mtx** extension. See [br_eval](api_docs/c_api/functions.md#br_eval) for more information.

* Mask Matrix- A mask matrix (or *mask*) is a binary matrix specified on page 14 of the [MBGC File Overview](misc/MBGC_file_overview.pdf) identifying the ground truth genuines and impostors of a corresponding *simmat*. Masks are identified with a **.mask** extension. See [br_make_mask](api_docs/c_api/functions.md#br_make_mask) and [br_combine_masks](api_docs/c_api/functions.md#br_combine_masks) for more information.

The evaluation harness is also accessible from the command line! Please see [-eval](api_docs/cl_api.md#eval), [-evalDetection](api_docs/cl_api.md#evaldetection), [-evalLandmarking](api_docs/cl_api.md#evallandmarking), [-evalClassification](api_docs/cl_api.md#evalclassification), [-evalClustering](api_docs/cl_api.md#evalclustering), or [-evalRegression](api_docs/cl_api.md#evalregression) for relevant information.

---

# Face Recognition

This tutorial gives an example on how to perform face recognition in OpenBR. OpenBR implements the 4SF[^2] algorithm to perform face recognition. Please read the the paper for more specific implementation details.

To start, lets run face recognition from the command line. Open the terminal and enter

    $ br -algorithm FaceRecognition \
        -compare ../data/MEDS/img/S354-01-t10_01.jpg ../data/MEDS/img/S354-02-t10_01.jpg \
         -compare ../data/MEDS/img/S354-01-t10_01.jpg ../data/MEDS/img/S386-04-t10_01.jpg

Easy enough? You should see results printed to terminal that look like

    $ Set algorithm to FaceRecognition
    $ Loading /usr/local/share/openbr/models/algorithms/FaceRecognition
    $ Loading /usr/local/share/openbr/models/transforms//FaceRecognitionExtraction
    $ Loading /usr/local/share/openbr/models/transforms//FaceRecognitionEmbedding
    $ Loading /usr/local/share/openbr/models/transforms//FaceRecognitionQuantization
    $Comparing ../data/MEDS/img/S354-01-t10_01.jpg and ../data/MEDS/img/S354-02-t10_01.jpg
    $ Enrolling ../data/MEDS/img/S354-01-t10_01.jpg to S354-01-t10_01r7Rv4W.mem
    $ 100.00%  ELAPSED=00:00:00  REMAINING=00:00:00  COUNT=1
    $ 100.00%  ELAPSED=00:00:00  REMAINING=00:00:00  COUNT=1
    $ 1.8812
    $ Comparing ../data/MEDS/img/S354-01-t10_01.jpg and ../data/MEDS/img/S386-04-t10_01.jpg
    $ Enrolling ../data/MEDS/img/S354-01-t10_01.jpg to S354-01-t10_01r7Rv4W.mem
    $ 100.00%  ELAPSED=00:00:00  REMAINING=00:00:00  COUNT=1
    $ 100.00%  ELAPSED=00:00:00  REMAINING=00:00:00  COUNT=1
    $ 0.571219

So what is 'FaceRecognition'? It's an abbrieviation to make running the algorithm easier. All of the algorithm abbreviations are located in ```openbr/plugins/core/algorithms.cpp```, please see the previous tutorial for an introduction to OpenBR's algorithm grammar.

It also possible to perform face recognition evaluation (**note:** this requires [R][R] to be installed)-

    $ br -algorithm FaceRecognition -path ../data/MEDS/img/ \
         -enroll ../data/MEDS/sigset/MEDS_frontal_target.xml target.gal \
         -enroll ../data/MEDS/sigset/MEDS_frontal_query.xml query.gal \
         -compare target.gal query.gal scores.mtx \
         -makeMask ../data/MEDS/sigset/MEDS_frontal_target.xml ../data/MEDS/sigset/MEDS_frontal_query.xml MEDS.mask \
         -eval scores.mtx MEDS.mask Algorithm_Dataset/FaceRecognition_MEDS.csv \
         -plot Algorithm_Dataset/FaceRecognition_MEDS.csv MEDS

face recognition search-

    $ br -algorithm FaceRecognition -enrollAll -enroll ../data/MEDS/img 'meds.gal;meds.csv[separator=;]'
    $ br -algorithm FaceRecognition -compare meds.gal ../data/MEDS/img/S001-01-t10_01.jpg match_scores.csv

and face recognition training-

    $ br -algorithm 'Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+(Grid(10,10)+SIFTDescriptor(12)+ByRow)/(Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,6,6)+Hist(59))+PCA(0.95)+Normalize(L2)+Dup(12)+RndSubspace(0.05,1)+LDA(0.98)+Cat+PCA(0.95)+Normalize(L1)+Quantize:NegativeLogPlusOne(ByteL1)' -train ../data/ATT/img FaceRecognitionATT

all right from the command line! The entire command line API is documented [here](api_docs/cl_api.md). It is a powerful tool for creating and testing new algorithms.

The command line isn't perfect for all situations however. So OpenBR exposes a [C++ API](api_docs/cpp_api.md) that can be embedded pretty much everywhere. Let's step through the example code at ```app/example/face_recognition.cpp``` and learn about using OpenBR as a library.

Our main function starts with-

        br::Context::initialize(argc, argv)

This is the first step in any OpenBR application, it initializes the global context.

    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("FaceRecognition");
    QSharedPointer<br::Distance> distance = br::Distance::fromAlgorithm("FaceRecognition");

Here, we split our algorithm into *enrollment* ([Transform](api_docs/cpp_api/transform/transform.md)::[fromAlgorithm](api_docs/cpp_api/transform/statics.md#fromalgorithm)) and *comparison* ([Distance](api_docs/cpp_api/distance/distance.md)::[fromAlgorithm](api_docs/cpp_api/distance/statics.md#fromalgorithm))

    br::Template queryA("../data/MEDS/img/S354-01-t10_01.jpg");
    br::Template queryB("../data/MEDS/img/S382-08-t10_01.jpg");
    br::Template target("../data/MEDS/img/S354-02-t10_01.jpg");

These lines create our [Templates](api_docs/cpp_api/template/template.md) for enrollment. They are just images loaded from disk. For this example queryA is the same person (different picture) as target and queryB is a different person.

    queryA >> *transform;
    queryB >> *transform;
    target >> *transform;

And now we enroll them! **>>** is a convienience operator for enrolling [Templates](api_docs/cpp_api/template/template.md) in [Transforms](api_docs/cpp_api/transform/transform.md). The enrollment is done in-place, which means the output overwrites the input. Our [Templates](api_docs/cpp_api/template/template.md) now store the results of enrollment.

    float comparisonA = distance->compare(target, queryA);
    float comparisonB = distance->compare(target, queryB);

Compare our query [Templates](api_docs/cpp_api/template/template.md) against the target [Template](api_docs/cpp_api/template/template.md). The result is a float.

    printf("Genuine match score: %.3f\n", comparisonA);
    printf("Impostor match score: %.3f\n", comparisonB);

Print out our results. You can see that comparisonA (between queryA and target) has a higher score then comparisonB, which is exactly what we expect!

    br::Context::finalize();

The last line in any OpenBR application has to be call to finalize. This shuts down OpenBR.

And that's it! Now you can embed face recognition into all of your applications.

---

# Age Estimation

Age Estimation is very similar in spirit to [Face Recognition](#face-recognition) and will be covered in far less detail.

To implement Age Estimation from the command line you can run

    $ br -algorithm AgeEstimation \
        -enroll ../data/MEDS/img/S354-01-t10_01.jpg ../data/MEDS/img/S001-01-t10_01.jpg metadata.csv

The results will be stored in metadata.csv under the key 'Age'. Remember from the [Face Recognition](#face-recognition) 'AgeEstimation' is just an abbreviation for the full algorithm description. The expanded version is stored in ```openbr/plugins/core/algorithms.cpp```.

The source code to run age estimation as an application is in ```app/examples/age_estimation.cpp```

---

# Gender Estimation

Gender Estimation is very similar in spirit to [Face Recognition](#face-recognition) and will be covered in far less detail.

To implement Gender Estimation from the command line you can run

    $ br -algorithm GenderEstimation \
        -enroll ../data/MEDS/img/S354-01-t10_01.jpg ../data/MEDS/img/S001-01-t10_01.jpg metadata.csv

The results will be stored in metadata.csv under the key 'Gender'. Remember from the [Face Recognition](#face-recognition) 'GenderEstimation' is just an abbreviation for the full algorithm description. The expanded version is stored in ```openbr/plugins/core/algorithms.cpp```.

The source code to run gender estimation as an application is in ```app/examples/gender_estimation.cpp```


[^1]: *Matthew Turk and Alex Pentland.*
    **Eigenfaces for recognition.**
    Journal of Cognitive Neuroscience, 71&#45;86, 1991 <!-- Don't know why I have to use the &#45; code instead of '-' -->
[^2]: *B. Klare.*
    **Spectrally sampled structural subspace features (4SF).**
    In Michigan State University Technical Report, MSUCSE-11-16, 2011
