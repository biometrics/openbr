
Welcome to OpenBR! Here we have a series of tutorials designed to get you up to speed on what OpenBR is, how it works, its command line interface, and the C API. These tutorials aren't meant to be completed in a specific order so feel free to hop around. If you need help, feel free to [contact us](index.md#help).

---

# OpenBR in 10 minutes or less!

This tutorial is meant to familiarize you with the ideas, objects and motivations behind OpenBR using some fun examples. **Note:** parts of this tutorial require a webcam.

OpenBR is a C++ library built on top of QT and OpenCV. It can either be used from the command line using the **br** application, or from interfacing with the [C API](#c-api). The command line is the easiest and fastest way to get started and this tutorial will use it for all of the examples.

First, make sure that OpenBR has been installed on your system using the steps described in the [installation section](#install.md). Then open up your terminal or command prompt and enter:

    $ br -gui -algorithm "Open+Show(false)" -enroll 0.webcam

If everything goes well your webcam should have opened up and is streaming. Look you are using OpenBR! Let's talk about what's happening in this command. OpenBR expects flags to be prepended by a *-* and arguments to follow the flags and be separated by spaces. Flags normally require a specific number of flags. All of the possible flags and their values are [documented here](#docs/cl_api.md). Let's step through the individual arguments and values. **-gui** is the flag that tells OpenBR to open up a GUI window. Take a look at the [GUI plugins](#docs/plugins/gui.md) for other plugins that require the **-gui** flag. **-algorithm** is one of the most important flags in OpenBR. It expects one argument called the *algorithm string*. This string determines the pipeline that images and metadata propagate through. Finally, **-enroll** reads files from disk and *enrolls* them into the image pipeline. It takes one input argument (0.webcam in our example) and an optional output argument. OpenBR has a range of formats that can be enrolled into algorithms, some examples include .jpg, .png, .csv, and .xml. .webcam tells OpenBR to enroll frames from the computers webcam.

Let's try a slightly more complicated example, after all OpenBR can do way more then just open webcams! Face detection is normally the first step in a [face recognition](#face recognition) algorithm. Let's do face detection in OpenBR. Open up the terminal again and enter:

    $ br -gui -algorithm "Open+Cvt(Gray)+Cascade(FrontalFace)+Draw+Show(false)" -enroll 0.webcam


It was built primarily as a platform for [face recognition](#face recognition) but has grown to do other things like [age recognition](#age recognition) and [gender recognition](#gender recognition). The different functionalities of OpenBR are specified by algorithms which are passed in as strings. The algorithm to do face recognition is

    $ Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(88,88,0.25,0.35)+<Mask+DenseSIFT/DenseLBP+DownsampleTraining(PCA(0.95),instances=1)+Normalize(L2)+Cat>+<Dup(12)+RndSubspace(0.05,1)+DownsampleTraining(LDA(0.98),instances=-2)+Cat+DownsampleTraining(PCA(768),instances=1)>+<Normalize(L1)+Quantize)>+SetMetadata(AlgorithmID,-1):Unit(ByteL1)

Woah, that's a lot! Face recognition is a pretty complicated process! We will break this whole string down in just a second, but first we can showcase one of the founding principles of OpenBR. In the face recognition algorithm there are a series of steps separated by +'s (and a few other symbols but they will all be explained); these steps are called plugins and they are the building blocks of all OpenBR algorithms. Each plugin is completely independent of all of the other plugins around it and each one can be swapped, inserted or removed at any time to form new algorithms. This makes it really easy to test new ideas as you come up with them!

So, now lets talk about the basics of algorithms in OpenBR. We know that algorithms are just a series of plugins joined together using the + symbol. What about the : symbol right at the end of the algorithm however? :'s separate the processing part of the algorithm (also called enrollment or generation), from the evaluation part. OpenBR actually has different types of plugins to handle each part. Plugins to the left are called [transforms](docs/cpp_api.md#transform) and plugins to the right are called [distances](docs/cpp_api.md#distance). Transforms operate on the images as they pass through the algorithm and distances compare (or find the distance between) the images as they finish, usually constructing a similarity matrix in the process.

This leads us on a small tangent to discuss how images are handled in OpenBR. OpenBR has two structures dedicated to handling data as it passes through an algorithm, [Files](docs/cpp_api.md#file) and [Templates](docs/cpp_api.md#template). Files handle metadata, the text and other information that can be associated with an image, and templates act as a container for images (we use OpenCV mats) and files. These templates are passed from transform to transform through the algorithm and are *transformed* (see, the name makes sense!) as they go.

Great, you now know how data is handled in OpenBR but we still have a lot to cover in that algorithm string! Next lets talk about all of the parentheses next to each plugin. Many plugins have parameters that can be set at runtime. For example, [Cvt](docs/plugins/imgproc.md#cvttransform) (short for convert) changes the colorspace of an image. For face recognition we want to change it to gray so we pass in Gray as the parameter to Cvt. Pretty simple right? Parameters can either be passed in in the order they appear in the code, or they can use a key-value pairing. Cvt(Gray) is equivalent to Cvt(ColorSpace=Gray), check out the docs for the properties of each plugin.

The last symbol we need to cover is <>. In OpenBR <> represents i/o. Transforms within these brackets will try and load their parameters from the hard drive if they can (they also still take parameters from the command line as you can see). Plugin i/o won't be covered here because it isn't critically important to making OpenBR work for you out of the gate. Stay tuned for a later tutorial on this.
