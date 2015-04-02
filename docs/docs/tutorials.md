# Tutorials

Learn OpenBR!

---

Welcome to OpenBR! Here we have a series of tutorials designed to get you up to speed on what OpenBR is, how it works, its command line interface, and the C API. These tutorials aren't meant to be completed in a specific order so feel free to hop around. If you need help, feel free to [contact us](index.md#help).

---

## OpenBR in 10 minutes or less!

This tutorial is meant to familiarize you with the ideas, objects and motivations behind OpenBR. If all you want to do is use some OpenBR applications or API calls the next tutorials might be more relevant to you.

OpenBR is a C++ library built on top of QT and OpenCV. It was built primarily as a platform for [face recognition](#face recognition) but has grown to do other things like [age recognition](#age recognition) and [gender recognition](#gender recognition). The different functionalities of OpenBR are specified by algorithms which are passed in as strings. The algorithm to do face recognition is

    $ Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(88,88,0.25,0.35)+<Mask+DenseSIFT/DenseLBP+DownsampleTraining(PCA(0.95),instances=1)+Normalize(L2)+Cat>+<Dup(12)+RndSubspace(0.05,1)+DownsampleTraining(LDA(0.98),instances=-2)+Cat+DownsampleTraining(PCA(768),instances=1)>+<Normalize(L1)+Quantize)>+SetMetadata(AlgorithmID,-1):Unit(ByteL1)

Woah, that's a lot! Face recognition is a pretty complicated process! We will break this whole string down in just a second, but first we can showcase one of the founding principles of OpenBR. In the face recognition algorithm there are a series of steps separated by +'s (and a few other symbols but they will all be explained); these steps are called plugins and they are the building blocks of all OpenBR algorithms. Each plugin is completely independent of all of the other plugins around it and each one can be swapped, inserted or removed at any time to form new algorithms. This makes it really easy to test new ideas as you come up with them!

So, now lets talk about the basics of algorithms in OpenBR. We know that algorithms are just a series of plugins joined together using the + symbol. What about the : symbol right at the end of the algorithm however? :'s separate the processing part of the algorithm (also called enrollment or generation), from the evaluation part. OpenBR actually has different types of plugins to handle each part. Plugins to the left are called [transforms](abstractions.md#transform) and plugins to the right are called [distances](abstractions.md#distance). Transforms operate on the images as they pass through the algorithm and distances compare (or find the distance between) the images as they finish, usually constructing a similarity matrix in the process.

This leads us on a small tangent to discuss how images are handled in OpenBR. OpenBR has two structures dedicated to handling data as it passes through an algorithm, [Files](abstractions.md#file) and [Templates](abstractions.md#template). Files handle metadata, the text and other information that can be associated with an image, and templates act as a container for images (we use OpenCV mats) and files. These templates are passed from transform to transform through the algorithm and are *transformed* (see, the name makes sense!) as they go.

Great, you now know how data is handled in OpenBR but we still have a lot to cover in that algorithm string! Next lets talk about all of the parentheses next to each plugin. Many plugins have parameters that can be set at runtime. For example, [Cvt](plugins.md#cvttransform) (short for convert) changes the colorspace of an image. For face recognition we want to change it to gray so we pass in Gray as the parameter to Cvt. Pretty simple right? Parameters can either be passed in in the order they appear in the code, or they can use a key-value pairing. Cvt(Gray) is equivalent to Cvt(ColorSpace=Gray), check out the docs for the properties of each plugin.

The last symbol we need to cover is <>. In OpenBR <> represents i/o. Transforms within these brackets will try and load their parameters from the hard drive if they can (they also still take parameters from the command line as you can see). Plugin i/o won't be covered here because it isn't critically important to making OpenBR work for you out of the gate. Stay tuned for a later tutorial on this. 
