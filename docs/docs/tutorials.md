
Welcome to OpenBR! Here we have a series of tutorials designed to get you up to speed on what OpenBR is, how it works, its command line interface, and the C API. These tutorials aren't meant to be completed in a specific order so feel free to hop around. If you need help, feel free to [contact us](index.md#help).

---

# Quick Start

This tutorial is meant to familiarize you with the ideas, objects and motivations behind OpenBR using some fun examples. **Note:** parts of this tutorial require a webcam.

OpenBR is a C++ library built on top of QT and OpenCV. It can either be used from the command line using the **br** application, or from interfacing with the [C API](docs/c_api.md). The command line is the easiest and fastest way to get started and this tutorial will use it for all of the examples.

First, make sure that OpenBR has been installed on your system using the steps described in the [installation section](install.md). Then open up your terminal or command prompt and enter:

    $ br -gui -algorithm "Show(false)" -enroll 0.webcam

If everything goes well your webcam should have opened up and is streaming. Look you are using OpenBR! Let's talk about what's happening in this command. OpenBR expects flags to be prepended by a *-* and arguments to follow the flags and be separated by spaces. Flags normally require a specific number of flags. All of the possible flags and their values are [documented here](docs/cl_api.md). Let's step through the individual arguments and values. **-gui** is the flag that tells OpenBR to open up a GUI window. Take a look at the [GUI plugins](docs/plugins/gui.md) for other plugins that require the **-gui** flag. **-algorithm** is one of the most important flags in OpenBR. It expects one argument called the *algorithm string*. This string determines the pipeline that images and metadata propagate through. Finally, **-enroll** reads files from disk and *enrolls* them into the image pipeline. It takes one input argument (0.webcam in our example) and an optional output argument. OpenBR has a range of formats that can be enrolled into algorithms, some examples include .jpg, .png, .csv, and .xml. .webcam tells OpenBR to enroll frames from the computers webcam.

Let's try a slightly more complicated example, after all OpenBR can do way more then just open webcams! Fire up the terminal again and enter:

    $ br -gui -algorithm "Cvt(Gray)+Show(false)" -enroll 0.webcam

Hey what happened? We took our normal BGR (OpenCV's alternative to RGB) image and converted it to a grayscale image. How did we do that? Simple, by adding "Cvt(Gray)" to the algorithm string. [Cvt](docs/plugins/imgproc.md#cvttransform), short for convert, is an example of an OpenBR *plugin*. [Show](docs/plugins/gui.md#showtransform) is a plugin as well. Every algorithm string in OpenBR is just a series of plugins joined together into a pipeline. In fact the **+** symbol is shorthand for a [Pipe](docs/plugins/core.md#pipetransform), another kind of OpenBR plugin. We specify **Gray** to [Cvt](docs/plugins/imgproc.md#cvttransform) as a runtime parameter to tell the plugin which color space to convert the image to. We also could have written **Cvt(HSV)** if we wanted to convert to the HSV color space or **Cvt(Luv)** if we wanted to convert to Luv. The arguments inside of the plugins are runtime parameters that can adjust the functionality. They can be provided as key-value pairs, **Cvt(Gray)** is equivalent to **Cvt(ColorSpace=Gray)**, or as keyless values. Make sure you are supplying parameters in the proper order if you are not using keys! Try and run the code with **Show(true)** and see how changing the parameters effect the output of the command (**Hint:** hit a key to cycle through the images).

Let's make this example a little bit more exciting and relevant to OpenBR's biometric roots. Face detection is normally the first step in a [face recognition](#face-recognition) algorithm. Let's do face detection in OpenBR. Back in the terminal enter:

    $ br -gui -algorithm "Cvt(Gray)+Cascade(FrontalFace)+Draw(lineThickness=3)+Show(false)" -enroll 0.webcam

You're webcam should be open again but this time a box should have appeared around your face! We added two new plugins to our string, [Cascade](docs/plugins/metadata.md#cascadetransform) and [Draw](docs/plugins/gui.md#drawtransform). Let's walk through this plugin by plugin and see how it works:

1. [Cvt(Gray)](docs/plugins/imgproc.md#cvttransform): Convert the image from BGR to grayscale. Grayscale is required for [Cascade](docs/plugins/metadata.md#cascadetransform) to work properly.
2. [Cascade(FrontalFace)](docs/plugins/metadata.md#cascadetransform): This is a wrapper on the OpenCV [Cascade Classification](http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html) framework. It detects frontal faces using the **FrontalFace** model.
3. [Draw(lineThickness=3)](docs/plugins/gui.md#drawtransform): Take the rects detected by [Cascade](docs/plugins/metadata.md#cascadetransform) and draw them onto the frame from the webcam. **lineThickness** determines the thickness of the drawn rect.
4. [Show(false)](docs/plugins/gui.md#showtransform): Show the image in a GUI window. **false** indicates the images should be shown in succession without waiting for a key press.

Pretty straightforward right? Each plugin completes one task and the passes the output on to the next plugin. You can pipe together as many plugins as you like as long as the output data from one can be the input data to the next. But wait! Output data? Input data? we haven't talked about data at all yet! How does OpenBR handle data? There are two objects that handle data is OpenBR; [Files](docs/cpp_api.md#file), which handle metadata, and [Templates](docs/cpp_api.md#template) which are containers for images and [Files](docs/cpp_api.md#file). Let's talk about [Files](docs/cpp_api.md#file) first. A file consists of file name, which is a path to a file on disk, and metadata which is a map of key-value pairs. The metadata can contain any textual information about the file. In the example above we use it to store the rectangles detected by [Cascade](docs/plugins/metadata.md#cascadetransform) and pass them along to [Draw](docs/plugins/metadata.md#drawtransform) for drawing. [Templates](docs/cpp_api.md#template) are containers for images, given as OpenCV [Mats](http://docs.opencv.org/modules/core/doc/basic_structures.html#mat), and [Files](docs/cpp_api.md#file). They can contain one image or a list of images. Plugins are either [Template](docs/cpp_api.md#template) in, [Template](docs/cpp_api.md#template) out or [TemplateList](docs/cpp_api.md#templatelist) in, [TemplateList](docs/cpp_api.md#templatelist) out. [TemplateLists](docs/cpp_api.md#templatelist) are, of course, just a list of [Templates](docs/cpp_api.md#template) which a few functions added for your convenience.

And there you go! You have gotten your quick start in OpenBR. We covered the [command line](docs/cl_api.md), plugins, and the key [data structures](docs/cpp_api.md) in OpenBR. If you want to learn more the next few tutorials will cover these fields with more depth. For even more information check out the [technical overviews](technical.md) and class documentation!

---

# Training Algorithms

---

# Face Recognition

---

# Age Estimation

---

# Gender Estimation
