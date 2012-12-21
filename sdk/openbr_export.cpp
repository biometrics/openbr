/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*!
 * \mainpage
 * \section overview Overview
 * \em OpenBR \cite openbr is a toolkit for biometric recognition and evaluation.
 * Supported use cases include training new recognition algorithms, interfacing with commercial systems, and measuring algorithm performance.
 * Free algorithms are also available for specific modalities including face recognition, face gender \& age estimation, face quality, and document classification.
 *
 * There are three modules users may interact with:
 * - \ref cli - \copybrief cli
 * - \ref c_sdk - \copybrief c_sdk
 * - \ref cpp_plugin_sdk - \copybrief cpp_plugin_sdk
 *
 * \section get_started Get Started
 * - \ref about
 * - \ref installation
 * - \ref examples
 * - \ref tutorial
 */

/*!
 * \page about About
 *
 * \em OpenBR was developed as a <a href="http://www.mitre.org/">MITRE</a> internal research project to facilitate prototyping new biometric algorithms and evaluating commercial systems.
 * It has been open sourced with the hope of providing a common framework for algorithm development and evaluation.
 *
 * OpenBR is written entirely in C/C++ and follows the <a href="http://semver.org">Semantic Versioning</a> convention for publishing releases.
 * The project uses the <a href="http://www.cmake.org">CMake</a> build system and depends on <a href="http://qt-project.org">Qt 4.8</a> and <a href="http://opencv.org">OpenCV 2.4.3</a>.
 * The \ref bee and the conventions established in the <a href="MBGC_file_overview.pdf">MBGC File Overview</a> for experimental setup are used for evaluating algorithm performance.
 *
 * - Developer mailing list: <a href="https://groups.google.com/forum/?fromgroups#!forum/openbr-dev">openbr-dev at googlegroups.com</a>
 * - Continuous integration server: <a href="http://my.cdash.org/index.php?project=OpenBR">CDash</a>
 *
 * \authors Josh Klontz \cite jklontz
 * \authors Mark Burge \cite mburge
 * \authors Brendan Klare \cite bklare
 * \authors E. Taborsky \cite mmtaborsky
 *
 * Please submit a pull request to add yourself to the authors list!
 */

/*!
 * \page installation Installation
 *
 * \section installation_from_source From Source
 * Installation from source is the recommended method for getting OpenBR on your machine.
 * If you need a little help getting started, choose from the list of build instructions for free C++ compilers below:
 * - \subpage windows_msvc
 * - \subpage windows_mingw
 * - \subpage osx_clang
 * - \subpage linux_gcc
 * - \subpage linux_icc
 *
 * \section installation_pre_built Pre-Built
 * Following your operating system's best practices is recommended for installing the pre-built compressed archives.
 *
 * However, for temporary evaluation, one simple configuration approach is:
 * \par Linux
\verbatim
$ cd bin
$ export LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH}
$ sudo ldconfig
$ sudo cp ../share/70-yubikey.rules /etc/udev/rules.d # Only needed if you were given a license dongle.
\endverbatim
 * \par OS X
\verbatim
$ cd bin
$ export DYLD_LIBRARY_PATH=../lib:${DYLD_LIBRARY_PATH}
$ export DYLD_FRAMEWORK_PATH=../lib:${DYLD_FRAMEWORK_PATH}
\endverbatim
 * \par Windows
 *  No configuration is necessary!
 *
 * \section installation_license_dongle License Dongle
 *  If you were given a USB License Dongle, then dongle must be in the computer in order to use the SDK.
 *  No configuration of the dongle is needed.
 *
 * \section installation_done Start Working
 * To test for successful installation:
\verbatim
$ cd bin/
$ br -help
\endverbatim
 */

/*!
 * \page windows_msvc Windows 7 - Visual Studio Express Edition 2012 - x64
 * -# Download and install <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-express-windows-desktop">Visual Studio 2012 Express Edition for Windows Desktop</a>
 *   -# If you need a program to mount ISO images then consider the free open source program <a href="http://wincdemu.sysprogs.org">WinCDEmu</a>.
 *   -# You will have to register with Microsoft after installation, but it's free.
 *   -# Grab any available <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-visual-studio-2012-update">Visual Studio Updates</a>.
 * -# Download and install <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe">CMake 2.8.10.2</a>
 *   -# During installation setup select "add CMake to PATH".
 * -# Download and unarchive <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">OpenCV 2.4.3</a>
 *   -# If you need a program to unarchive tarballs then consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a>.
 *   -# Copy the "OpenCV-2.4.3" folder to "C:\" and rename it "OpenCV-2.4.3-msvc2012".
 *   -# From the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt" and enter:
 *   \code
 *   $ cd C:\OpenCV-2.4.3-msvc2012
 *   $ mkdir build
 *   $ cd build
 *   $ cmake -G "Visual Studio 11 Win64" -D WITH_FFMPEG=OFF ..
 *   \endcode
 *   -# Open "C:\OpenCV-2.4.3-msvc2012\build\OpenCV.sln"
 *     -# Under the "BUILD" menu, select "Build Solution".
 *     -# Switch from "Debug" to "Release" and repeat the above step.
 * -# Download and install <a href="http://releases.qt-project.org/qt4/source/qt-win-opensource-4.8.4-vs2010.exe">Qt 4.8.4</a>
 *   -# From the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt" and enter:
 *   \code
 *   $ cd C:\Qt\4.8.4
 *   $ configure.exe -platform win32-msvc2012 -no-webkit
 *   $ nmake
 *   \endcode
 *     -# Select the Open Source Edition.
 *     -# Accept the license offer.
 *     -# configure.exe will take several minutes to finish.
 *     -# nmake will take several hours to finish.
 * -# Create a <a href="github.com">GitHub</a> account and follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
 *   -# Launch "Git Bash" and clone OpenBR:
 *   \code
 *   $ cd /c
 *   $ git clone https://github.com/biometrics/openbr.git
 *   \endcode
 * -# Finally time to actually build OpenBR:
 *   -# From the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt" and enter:
 *   \code
 *   $ cd C:\mm
 *   $ mkdir build-msvc2012
 *   $ cd build-msvc2012
 *   $ cmake -G "NMake Makefiles" -D OpenCV_DIR="C:\OpenCV-2.4.3-msvc2012\build" -D QT_QMAKE_EXECUTABLE="C:\Qt\4.8.4\bin\qmake" ..
 *   $ nmake
 *   \endcode
 * -# To package OpenBR:
 *   -# From the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt" and enter:
 *   \code
 *   $ cd C:\mm
 *   $ nmake package
 *   \endcode
 */

/*!
 * \page windows_mingw Windows 7 - MingGW-w64 2.0 - x64
 * -# <a href="http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/rubenvb/gcc-4.7-release/x86_64-w64-mingw32-gcc-4.7.2-release-win64_rubenvb.7z/download">Download MinGW-w64 GCC 4.7.2</a> and unarchive.
 *   -# Use the free open source program <a href="http://www.7-zip.org/">7-Zip</a> to unarchive.
 *   -# Copy "x86_64-w64-mingw32-gcc-4.7.2-release-win64_rubenvb\mingw64" to "C:\".
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe">Download CMake 2.8.10.2</a> and install.
 *   -# During installation setup select "add CMake to PATH".
 * -# <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">Download OpenCV 2.4.3</a> and unarchive.
 *   -# If you need a program to unarchive tarballs then consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a>.
 *   -# Copy the "OpenCV-2.4.3" folder to "C:\" and rename it "OpenCV-2.4.3-mingw64".
 *   -# Double-click "C:\mingw64\mingw64env.cmd" and enter:
 *   \code
 *   $ cd C:\OpenCV-2.4.3-mingw64
 *   $ mkdir build
 *   $ cd build
 *   $ cmake -G "MinGW Makefiles" -D WITH_FFMPEG=OFF -D CMAKE_BUILD_TYPE=Release ..
 *   $ mingw32-make
 *   \endcode
 * -# <a href="http://releases.qt-project.org/qt4/source/qt-everywhere-opensource-src-4.8.4.zip">Download Qt 4.8.4</a> and unzip.
 *   -# Copy "qt-everywhere-opensource-src-4.8.4" to "C:\".
 *   -# Double-click "C:\mingw64\mingw64env.cmd" and enter:
 *   \code
 *   $ cd C:\qt-everywhere-opensource-src-4.8.4
 *   $ configure.exe
 *   $ mingw32-make
 *   \endcode
 *     -# Select the Open Source Edition.
 *     -# Accept the license offer.
 *     -# configure.exe will take several minutes to finish.
 *     -# mingw32-make will take several hours to finish.
 * -# Create a <a href="github.com">GitHub</a> account and follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
 *   -# Launch "Git Bash" and clone OpenBR:
 *   \code
 *   $ git clone https://github.com/biometrics/openbr.git
 *   $ cd /c
 *   \endcode
 * -# Finally time to build OpenBR:
 *   -# Double-click "C:\mingw64\mingw64env.cmd" and enter:
 *   \code
 *   $ cd C:\mm
 *   $ mkdir build-mingw64
 *   $ cd build-mingw64
 *   $ cmake -G "MinGW Makefiles" -D CMAKE_RC_COMPILER="C:/mingw64/bin/windres.exe" -D OpenCV_DIR="C:\OpenCV-2.4.3-mingw64\build" -D QT_QMAKE_EXECUTABLE="C:\qt-everywhere-opensource-src-4.8.4\bin\qmake" ..
 *   $ mingw32-make
 *   \endcode
 * -# To package OpenBR:
 *  -# <a href="http://prdownloads.sourceforge.net/nsis/nsis-2.46-setup.exe?download">Download NSIS 2.46</a> and install.
 *  -# Double-click "C:\mingw64\mingw64env.cmd" and enter:
 *  \code
 *  $ cd C:\mm\build-mingw64
 *  $ mingw32-make package
 *  \endcode
 */

/*!
 * \page osx_clang OS X Mountain Lion - Clang/LLVM 3.1 - x64
 * -# Download and install the latest "Xcode" and "Command Line Tools" from the <a href="https://developer.apple.com/downloads/index.action#">Apple Developer Downloads</a> page.
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2.tar.gz">Download CMake 2.8.10.2</a> and install:
 * \code
 * $ cd ~/Downloads
 * $ tar -xf cmake-2.8.10.2.tar.gz
 * $ cd cmake-2.8.10.2
 * $ ./configure
 * $ make
 * $ sudo make install
 * \endcode
 * -# <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">Download OpenCV 2.4.3</a> and install:
 * \code
 * $ cd ~/Downloads
 * $ tar -xf OpenCV-2.4.3.tar.bz2
 * $ cd OpenCV-2.4.3
 * $ mkdir build
 * $ cd build
 * $ cmake ..
 * $ make
 * $ sudo make install
 * \endcode
 * -# <a href="http://releases.qt-project.org/qt4/source/qt-mac-opensource-4.8.4.dmg">Download Qt 4.8.4</a> and install.
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>, then clone:
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * \endcode
 * -# Finally time to build OpenBR:
 * \code
 * $ cd mm
 * $ mkdir build
 * $ cd build
 * $ cmake ..
 * $ make
 * \endcode
 * -# To package OpenBR <a href="https://developer.apple.com/downloads/index.action#">Download Auxilary Tools for Xcode</a>, drag "PackageMaker" to "Applications", then package:
 * \code
 * $ cd mm/build
 * $ sudo make package # PackageMaker requires sudo
 * \endcode
 */

/*!
 * \page linux_gcc Ubuntu 12.04 LTS - GCC 4.6.3 - x64
 * -# Install GCC 4.6.3:
 * \code
 * $ sudo apt-get update
 * $ sudo apt-get install build-essential
 * \endcode
 * -# Install CMake 2.8.7:
 * \code
 * $ sudo apt-get install cmake
 * \endcode
 * -# <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">Download OpenCV 2.4.3</a> and install:
 * \code
 * $ cd ~/Downloads
 * $ tar -xf OpenCV-2.4.3.tar.bz2
 * $ cd OpenCV-2.4.3
 * $ mkdir build
 * $ cd build
 * $ cmake -D CMAKE_BUILD_TYPE=Release ..
 * $ make
 * $ sudo make install
 * \endcode
 * -# Install Qt 4.8.1:
 * \code
 * $ sudo apt-get install libqt4-dev
 * \endcode
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>, then clone:
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * \endcode
 * -# Finally time to build OpenBR:
 * \code
 * $ cd mm
 * $ mkdir build
 * $ cd build
 * $ cmake ..
 * $ make
 * \endcode
 * -# To package OpenBR:
 * \code
 * $ cd mm/build
 * $ make package
 * \endcode
 */

/*!
 * \page linux_icc Ubuntu 12.04 LTS - Intel C++ Studio XE 2013 - x64
 * -# <a href="http://software.intel.com/en-us/non-commercial-software-development">Download Intel C++ Studio XE 2013</a> and install.
 * -# Install CMake 2.8.7:
 * \code
 * $ sudo apt-get install cmake
 * \endcode
 * -# <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">Download OpenCV 2.4.3</a> and install:
 * \code
 * $ cd ~/Downloads
 * $ tar -xf OpenCV-2.4.3.tar.bz2
 * $ cd OpenCV-2.4.3
 * $ mkdir build
 * $ cd build
 * $ cmake -D CMAKE_BUILD_TYPE=Release ..
 * $ make
 * $ sudo make install
 * \endcode
 * -# Install Qt 4.8.1:
 * \code
 * $ sudo apt-get install libqt4-dev
 * \endcode
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>, then clone:
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * \endcode
 */

/*!
 * \page examples Examples
 * \brief Source code example applications and their equivalent \ref cli expressions.
 *
 * Many examples make heavy use of the \ref bee and the conventions established in the <a href="MBGC_file_overview.pdf">MBGC File Overview</a> for experimental setup.
 * - \ref compare_faces
 * - \ref compare_face_galleries
 * - \ref evaluate_face_recognition
 *
 * \section compare_faces Compare Faces
 * \snippet app/examples/compare_faces.cpp compare_faces
 * \section compare_face_galleries Compare Face Galleries
 * \snippet app/examples/compare_face_galleries.cpp compare_face_galleries
 * \section evaluate_face_recognition Evaluate Face Recognition
 * \snippet app/examples/evaluate_face_recognition.cpp evaluate_face_recognition
 */

/*!
 * \page tutorial Tutorial
 * \brief An end-to-end example covering experimental setup, algorithm development, and performance evaluation on the <a href="http://yann.lecun.com/exdb/mnist/
">MNIST Handwritten Digits Dataset</a>.
 *
 * Under construction, please check back soon!
 */

/*!
 * \page installing_r Installing R
 * The \c br reporting framework requires a valid \c R installation in order to generate performance figures. Please follow the instructions below.
 * -# <a href="http://watson.nci.nih.gov/cran_mirror/">Download and Install R</a>
 * -# Run \c R
 *   -# Enter the command:
 *    \code install.packages(c("ggplot2", "gplots", "reshape", "scales")) \endcode
 *   -# When prompted, select a \c mirror near you.
 *   -# Wait for the package installation to complete.
 * -# Exit \c R
 * \note Installation process requires internet access.
 */

/*!
 * \page bee Biometric Evaluation Environment
 * \brief The <i>Biometric Evaluation Environment</i> (BEE) is a <a href="http://www.nist.gov/index.html">NIST</a> standard for evaluating biometric algorithms.
 *
 * OpenBR implements the following portions of the BEE specification:
 *
 * \section sigset Signature Set
 * A signature set (or \em sigset) is a br::Gallery compliant \c XML file-list specified on page 9 of <a href="MBGC_file_overview.pdf#page=9">MBGC File Overview</a> and implemented in xmlGallery.
 * Sigsets are identified with a <tt>.xml</tt> extension.
 *
 * \section simmat Similarity Matrix
 * A similarity matrix (or \em simmat) is a br::Output compliant binary score matrix specified on page 12 of <a href="MBGC_file_overview.pdf#page=12">MBGC File Overview</a> and implemented in mtxOutput.
 * Simmats are identified with a <tt>.mtx</tt> extension.
 * \see br_eval
 *
 * \section mask Mask Matrix
 * A mask matrix (or \em mask) is a binary matrix specified on page 14 of <a href="MBGC_file_overview.pdf#page=14">MBGC File Overview</a> identifying the ground truth genuines and impostors of a corresponding \ref simmat.
 * Masks are identified with a <tt>.mask</tt> extension.
 * \see br_make_mask br_combine_masks
 */

/*!
 * \page managed_return_value Managed Return Value
 * Memory for the returned value is managed internally and guaranteed until the next call to this function.
 */
