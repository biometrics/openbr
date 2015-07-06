## Overview

OpenBR is a framework for investigating new modalities, improving existing algorithms, interfacing with commercial systems, measuring recognition performance, and deploying automated biometric systems.
The project is designed to facilitate rapid algorithm prototyping, and features a mature core framework, flexible plugin system, and support for open and closed source development.
Off-the-shelf algorithms are also available for specific modalities including [face recognition](tutorials.md#face-recognition), [age estimation](tutorials.md#age-estimation), and [gender estimation](tutorials.md#gender-estimation). Please see the [Tutorials](tutorials.md) section for more information.

OpenBR originated within The MITRE Corporation from a need to streamline the process of prototyping new algorithms.
The project was later published as open source software under the [Apache 2](http://www.apache.org/licenses/LICENSE-2.0.html) license and is *free for academic and commercial use*.

Please read [our paper](http://openbiometrics.org/publications/klontz2013open.pdf) for more information about OpenBR and kindly cite in your own work:

> J. Klontz, B. Klare, S. Klum, M. Burge, A. Jain, “Open Source Biometric Recognition”, Biometrics: Theory, Applications and Systems, 2013.

---

<figure id="abstraction" style="display:block; margin: 0 auto;">
  <img src="img/abstraction.svg">
  <figcaption>The two principal software artifacts are the shared library 'openbr' and command line application 'br'.</figcaption>
</figure>

---

## Install

OpenBR is supported on multiple operating systems. Please select yours from the list below for installation instructions.

* [Linux](install.md#linux)
* [Mac OSX](install.md#osx)
* [Windows](install.md#windows)
* [Raspian](install.md#raspian)

---

## Tutorials

We recommend that first-time users of the OpenBR framework look at the following tutorials before moving on to the more in-depth documentation below.

* [Quick Start](tutorials.md#quick-start)
* [Algorithms in OpenBR](tutorials.md#algorithms-in-openbr)
* [Training Algorithms in OpenBR](tutorials.md#training-algorithms)
* [The Evaluation Harness](tutorials.md#the-evaluation-harness)
* [Face Recognition](tutorials.md#face-recognition)
* [Age Estimation](tutorials.md#age-estimation)
* [Gender Estimation](tutorials.md#gender-estimation)

---

## Documentation

Here is the complete documentation for OpenBR. Enjoy!

* [The C API](api_docs/c_api.md)
* [The C++ Plugin API](api_docs/cpp_api.md)
* [The Command Line Interface](api_docs/cl_api.md)

---

## Help

Still have questions? Please reach out to us on our [developer mailing list](https://groups.google.com/forum/?fromgroups#!forum/openbr-dev).
