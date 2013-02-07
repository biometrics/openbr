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
 * OpenBR \cite openbr is a toolkit for biometric recognition and evaluation.
 * Supported use cases include training new recognition algorithms, interfacing with commercial systems, and measuring algorithm performance.
 * Free algorithms are also available for specific modalities including face recognition, face gender \& age estimation, face quality, and document classification.
 *
 * There are three modules users may interact with:
 * - \ref cli - \copybrief cli
 * - \ref c_sdk - \copybrief c_sdk
 * - \ref cpp_plugin_sdk - \copybrief cpp_plugin_sdk
 *
 * \section get_started Get Started
 * - \ref installation
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
 * \brief Installation from source with Visual Studio.
 *
 * -# Download and install <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-express-windows-desktop">Visual Studio 2012 Express Edition for Windows Desktop</a>
 *   -# If you need a program to mount ISO images then consider the free open source program <a href="http://wincdemu.sysprogs.org">WinCDEmu</a>.
 *   -# You will have to register with Microsoft after installation, but it's free.
 *   -# Grab any available <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-visual-studio-2012-update">Visual Studio Updates</a>.
 * -# Download and install <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe">CMake 2.8.10.2</a>
 *   -# During installation setup select "add CMake to PATH".
 * -# Download and unarchive <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">OpenCV 2.4.3</a>
 *   -# If you need a program to unarchive tarballs then consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a>.
 *   -# Copy the "OpenCV-2.4.3" folder to "C:\".
 *   -# From the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt" and enter:
 *   \code
 *   $ cd C:\OpenCV-2.4.3
 *   $ mkdir build-msvc2012
 *   $ cd build-msvc2012
 *   $ cmake -G "NMake Makefiles" -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_FFMPEG=OFF -D CMAKE_BUILD_TYPE=Debug ..
 *   $ nmake
 *   $ nmake install
 *   $ cmake -D CMAKE_BUILD_TYPE=Release ..
 *   $ nmake
 *   $ nmake install
 *   $ nmake clean
 *   \endcode
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
 *   $ cd openbr
 *   $ git submodule init
 *   $ git submodule update
 *   \endcode
 * -# Finally time to actually build OpenBR:
 *   -# From the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt" and enter:
 *   \code
 *   $ cd C:\openbr
 *   $ mkdir build-msvc2012
 *   $ cd build-msvc2012
 *   $ cmake -G "CodeBlocks - NMake Makefiles" -D CMAKE_PREFIX_PATH="C:/OpenCV-2.4.3/build-msvc2012/install" -D QT_QMAKE_EXECUTABLE="C:/Qt/4.8.4/bin/qmake" -D BR_INSTALL_DEPENDENCIES=ON -D CMAKE_BUILD_TYPE=Release ..
 *   $ nmake
 *   $ nmake package
 *   \endcode
 */

/*!
 * \page windows_mingw Windows 7 - MinGW-w64 2.0 - x64
 * \brief Installation from source with MinGW-w64.
 *
 * -# <a href="http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/rubenvb/gcc-4.7-release/x86_64-w64-mingw32-gcc-4.7.2-release-win64_rubenvb.7z/download">Download MinGW-w64 GCC 4.7.2</a> and unarchive.
 *   -# Use the free open source program <a href="http://www.7-zip.org/">7-Zip</a> to unarchive.
 *   -# Copy "x86_64-w64-mingw32-gcc-4.7.2-release-win64_rubenvb\mingw64" to "C:\".
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe">Download CMake 2.8.10.2</a> and install.
 *   -# During installation setup select "add CMake to PATH".
 * -# <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download">Download OpenCV 2.4.3</a> and unarchive.
 *   -# If you need a program to unarchive tarballs then consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a>.
 *   -# Copy the "OpenCV-2.4.3" folder to "C:\".
 *   -# Double-click "C:\mingw64\mingw64env.cmd" and enter:
 *   \code
 *   $ cd C:\OpenCV-2.4.3
 *   $ mkdir build-mingw64
 *   $ cd build-mingw64
 *   $ cmake -G "MinGW Makefiles" -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_FFMPEG=OFF -D CMAKE_BUILD_TYPE=Debug ..
 *   $ mingw32-make
 *   $ mingw32-make install
 *   $ cmake -D CMAKE_BUILD_TYPE=Release
 *   $ mingw32-make
 *   $ mingw32-make install
 *   $ mingw32-make clean
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
 *   $ cd /c
 *   $ git clone https://github.com/biometrics/openbr.git
 *   $ cd openbr
 *   $ git submodule init
 *   $ git submodule update
 *   \endcode
 * -# Finally time to build OpenBR:
 *   -# Double-click "C:\mingw64\mingw64env.cmd" and enter:
 *   \code
 *   $ cd C:\openbr
 *   $ mkdir build-mingw64
 *   $ cd build-mingw64
 *   $ cmake -G "CodeBlocks - MinGW Makefiles" -D CMAKE_RC_COMPILER="C:/mingw64/bin/windres.exe" -D CMAKE_PREFIX_PATH="C:/OpenCV-2.4.3/build-mingw64/install" -D QT_QMAKE_EXECUTABLE="C:/qt-everywhere-opensource-src-4.8.4/bin/qmake" -D BR_INSTALL_DEPENDENCIES=ON -D CMAKE_BUILD_TYPE=Release ..
 *   $ mingw32-make
 *   $ mingw32-make package
 *  \endcode
 */

/*!
 * \page osx_clang OS X Mountain Lion - Clang/LLVM 3.1 - x64
 * \brief Installation from source with Clang.
 *
 * -# Download and install the latest "Xcode" and "Command Line Tools" from the <a href="https://developer.apple.com/downloads/index.action#">Apple Developer Downloads</a> page.
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2.tar.gz">Download CMake 2.8.10.2</a> and install:
 * \code
 * $ cd ~/Downloads
 * $ tar -xf cmake-2.8.10.2.tar.gz
 * $ cd cmake-2.8.10.2
 * $ ./configure
 * $ make
 * $ sudo make install
 * $ cd ..
 * $ rm -r cmake-2.8.10.2
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
 * $ cd ../..
 * $ rm -r OpenCV-2.4.3
 * \endcode
 * -# <a href="http://releases.qt-project.org/qt4/source/qt-mac-opensource-4.8.4.dmg">Download Qt 4.8.4</a> and install.
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>, then clone:
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * $ cd openbr
 * $ git submodule init
 * $ git submodule update
 * \endcode
 * -# Finally time to build OpenBR:
 * \code
 * $ cd openbr
 * $ mkdir build
 * $ cd build
 * $ cmake ..
 * $ make
 * $ make install
 * \endcode
 */

/*!
 * \page linux_gcc Ubuntu 12.04 LTS - GCC 4.6.3 - x64
 * \brief Installation from source with GCC.
 *
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
 * $ cd ../..
 * $ rm -r OpenCV-2.4.3
 * \endcode
 * -# Install Qt 4.8.1:
 * \code
 * $ sudo apt-get install libqt4-dev
 * \endcode
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>, then clone:
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * $ cd openbr
 * $ git submodule init
 * $ git submodule update
 * \endcode
 * -# Finally time to build OpenBR:
 * \code
 * $ cd openbr
 * $ mkdir build
 * $ cd build
 * $ cmake ..
 * $ make
 * $ make install
 * \endcode
 */

/*!
 * \page linux_icc Ubuntu 12.04 LTS - Intel C++ Studio XE 2013 - x64
 * \brief Installation from source with ICC.
 *
 * -# Assuming you meet the eligibility requirements, <a href="http://software.intel.com/en-us/non-commercial-software-development">Download Intel C++ Studio XE 2013</a> and install.
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
 * $ cd ../..
 * $ rm -r OpenCV-2.4.3
 * \endcode
 * -# Install Qt 4.8.1:
 * \code
 * $ sudo apt-get install libqt4-dev
 * \endcode
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>, then clone:
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * $ cd openbr
 * $ git submodule init
 * $ git submodule update
 * \endcode
 * -# Finally time to build OpenBR:
 * \code
 * $ cd openbr
 * $ mkdir build-icc
 * $ cd build-icc
 * $ cmake -DCMAKE_C_COMPILER=/opt/intel/bin/icc -DCMAKE_CXX_COMPILER=/opt/intel/bin/icpc ..
 * $ make
 * $ make install
 * \endcode
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
