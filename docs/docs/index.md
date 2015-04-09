# OpenBR ![Overview](img/openbr_48x48.png)

<p id=tagline>Open source, industry quality biometrics.</p>

---

## Overview

OpenBR \cite klontz2013open is a framework for investigating new modalities, improving existing algorithms, interfacing with commercial systems, measuring recognition performance, and deploying automated biometric systems.
The project is designed to facilitate rapid algorithm prototyping, and features a mature core framework, flexible plugin system, and support for open and closed source development.
Off-the-shelf algorithms are also available for specific modalities including [face recognition](tutorials.md#face-recognition), [age estimation](tutorials.md#age-estimation), and [gender estimation](tutorials.md#gender-estimation). Please see the [Tutorials](tutorials.md) section for more information.

OpenBR originated within The MITRE Corporation from a need to streamline the process of prototyping new algorithms.
The project was later published as open source software under the [Apache 2](http://www.apache.org/licenses/LICENSE-2.0.html) license and is *free for academic and commercial use*.

![Overview](img/abstraction.svg)

*The two principal software artifacts are the shared library 'openbr' and command line application 'br'.*

---

## Getting Started

OpenBR is supported on multiple operating systems. Please select yours from the list below for installation instructions. Happy Hacking!

* [Linux](install.md#linux)
* [Mac OSX](install.md#osx)
* [Windows](install.md#windows)
* [Raspian](install.md#raspian)

---

## The Basics

We have created a few tutorials to help teach you the basic principles of the OpenBR system. If this is your first time using OpenBR you should look at these before moving on to the more technical descriptions below.

* [Quick Start](tutorials.md#quick-start)
* [Training in OpenBR](tutorials.md#training-algorithms)
* [Face Recognition](tutorials.md#face-recognition)
* [Age Estimation](tutorials.md#age-estimation)
* [Gender Estimation](tutorials.md#gender-estimation)

---

## Additional Materials

OpenBR has several parts. Below are several more technical descriptions of important parts of the system. Enjoy!

* [Algorithms in OpenBR](technical.md#algorithms-in-openbr)
* [The C API](docs/c_api.md)
* [The Command Line Interface](docs/cl_api.md)
* [The C++ Plugin API](docs/cpp_api.md)
* [The Evaluation Harness](technical.md#the-evaluation-harness)

---

## Help

Still have questions? Please reach out to us on our <a href="https://groups.google.com/forum/?fromgroups#!forum/openbr-dev">developer mailing list</a> or our <a href="http://webchat.freenode.net/?channels=openbr">IRC channel</a>
