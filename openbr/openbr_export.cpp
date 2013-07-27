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
 * OpenBR \cite klontz2013open is a framework for investigating new modalities, improving existing algorithms, interfacing with commercial systems, measuring recognition performance, and deploying automated biometric systems.
 * The project is designed to facilitate rapid algorithm prototyping, and features a mature core framework, flexible plugin system, and support for open and closed source development.
 * Off-the-shelf algorithms are also available for specific modalities including \ref cpp_face_recognition, \ref cpp_age_estimation, and \ref cpp_gender_estimation.
 *
 * OpenBR originated within The MITRE Corporation from a need to streamline the process of prototyping new algorithms.
 * The project was later published as open source software under the <a href="http://www.apache.org/licenses/LICENSE-2.0.html">Apache 2</a> license and is <i>free for academic and commercial use</i>.
 *
 * \image html "abstraction.svg" "The two principal software artifacts are the shared library 'openbr' and command line application 'br'."
 *
 * \section get_started Get Started
 * - \ref introduction - A high-level technical overview of OpenBR.
 * - \ref installation - A hacker's guide to building, editing, and running OpenBR.
 * - \ref qmake_integration - Add OpenBR to your Qt <tt>.pro</tt> project.
 *
 * \section learn_more Learn More
 * - \ref algorithm_grammar - How algorithms are constructed from string descriptions.
 * - \ref cli - Command line wrapper of the \ref c_sdk.
 * - \ref c_sdk - High-level API for running algorithms and evaluating results.
 * - \ref cpp_plugin_sdk - Plugin API for extending OpenBR functionality.
 * - \ref bee - A <a href="http://www.nist.gov/index.html">NIST</a> standard for evaluating biometric algorithms.
 */

/*!
 * \page introduction Introduction
 * \brief A high-level technical overview of OpenBR.
 *
 * We strongly encourage users new to OpenBR to read our <a href="www.openbiometrics.org/publications/klontz2013open.pdf"><b>publication</b></a> for an introduction to the core concepts.
 * Researchers incorporating OpenBR into their own work are kindly requested to cite this paper.
 */

/*!
 * \page installation Installation
 * \brief A hacker's guide to building, editing, and running OpenBR.
 *
 * \section installation_from_source From Source
 * Installation from source is the recommended method for getting OpenBR on your machine.
 * If you need a little help getting started, choose from the list of build instructions for free C++ compilers below:
 * - \subpage windows_msvc
 * - \subpage windows_mingw
 * - \subpage osx_clang
 * - \subpage linux_all
 *
 * \section installation_from_binary From Binary
 * Pre-compiled releases are not currently provided, but they can be built from source using the instructions above.
 * Generally you should follow your operating system's best practices for installing a binary package.
 * However, for temporary evaluation, one simple configuration approach is:
 *
 * \par Linux
\verbatim
$ cd bin
$ export LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH}
$ sudo ldconfig
$ sudo cp ../share/openbr/70-yubikey.rules /etc/udev/rules.d # Only needed if you were given a license dongle.
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
 *  In the unlikely event that you were given a USB License Dongle, then dongle must be in the computer in order to use the SDK.
 *  No configuration of the dongle is needed.
 *
 * \section installation_done Start Working
 * To test for successful installation:
\verbatim
$ cd bin
$ br -help
\endverbatim
 */

/*!
 * \page windows_msvc Windows 7 - Visual Studio Express Edition 2012 - x64
 * -# <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-express-windows-desktop">Download Visual Studio 2012 Express Edition for Windows Desktop</a> and install.
 *  -# Consider the free open source program <a href="http://wincdemu.sysprogs.org">WinCDEmu</a> if you need a program to mount ISO images.
 *  -# You will have to register with Microsoft after installation, but it's free.
 *  -# Grab any available <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-visual-studio-2012-update">Visual Studio Updates</a>.
 *  -# Download and install <a href="http://msdn.microsoft.com/en-us/windows/hardware/hh852363.aspx">Windows 8 SDK</a>.
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe">Download and Install CMake 2.8.10.2</a>
 *  -# During installation setup select "add CMake to PATH".
 * -# <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.5/opencv-2.4.5.tar.gz">Download OpenCV 2.4.5</a>
 *  -# Consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a> if you need a program to unarchive tarballs.
 *  -# Move the "opencv-2.4.5" folder to "C:\".
 *  -# Open "VS2012 x64 Cross Tools Command Prompt" (from the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt") and enter:
 *  \code
 *  $ cd C:\opencv-2.4.5
 *  $ mkdir build-msvc2012
 *  $ cd build-msvc2012
 *  $ cmake -G "NMake Makefiles" -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=OFF -DCMAKE_BUILD_TYPE=Debug ..
 *  $ nmake
 *  $ nmake install
 *  $ cmake -DCMAKE_BUILD_TYPE=Release ..
 *  $ nmake
 *  $ nmake install
 *  $ nmake clean
 *  \endcode
 * -# <a href="http://download.qt-project.org/official_releases/qt/5.0/5.0.2/qt-windows-opensource-5.0.2-msvc2012_64-x64-offline.exe">Download and Install Qt 5.0.2</a>
 * -# Create a <a href="github.com">GitHub</a> account and follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
 *  -# Launch "Git Bash" from the Desktop and clone OpenBR:
 *  \code
 *  $ cd /c
 *  $ git clone https://github.com/biometrics/openbr.git
 *  $ cd openbr
 *  $ git submodule init
 *  $ git submodule update
 *  \endcode
 * -# Build OpenBR!
 *  -# From the VS2012 x64 Cross Tools Command Prompt:
 *  \code
 *  $ cd C:\openbr
 *  $ mkdir build-msvc2012
 *  $ cd build-msvc2012
 *  $ cmake -G "CodeBlocks - NMake Makefiles" -DCMAKE_PREFIX_PATH="C:/openCV-2.4.5/build-msvc2012/install;C:/Qt/Qt5.0.2/5.0.2/msvc2012_64" -DCMAKE_INSTALL_PREFIX="./install" -DBR_INSTALL_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release ..
 *  $ nmake
 *  $ nmake install
 *  \endcode
 *  -# Check out the "install" folder.
 * -# Hack OpenBR!
 *  -# From the VS2012 x64 Cross Tools Command Prompt:
 *  \code
 *  $ C:\Qt\Qt5.0.2\Tools\QtCreator\bin\qtcreator.exe
 *  \endcode
 *  -# From the Qt Creator "Tools" menu select "Options..."
 *  -# Under "Kits" select "Desktop (default)"
 *  -# For "Compiler:" select "Microsoft Visual C++ Compiler 11.0 (x86_amd64)" and click "OK"
 *  -# From the Qt Creator "File" menu select "Open File or Project...".
 *  -# Select "C:\openbr\CMakeLists.txt" then "Open".
 *  -# If prompted for the location of CMake, enter "C:\Program Files (x86)\CMake 2.8\bin\cmake.exe".
 *  -# Browse to your pre-existing build directory "C:\openbr\build-msvc2012" then select "Next".
 *  -# Select "Run CMake" then "Finish".
 *  -# You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.
 * -# (Optional) Package OpenBR!
 *  -# From the VS2012 x64 Cross Tools Command Prompt:
 *  \code
 *  $ cd C:\openbr\build-msvc2012
 *  $ cpack -G ZIP
 *  \endcode
 */

/*!
 * \page windows_mingw Windows 7 - MinGW-w64 2.0 - x64
 * -# <a href="http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/rubenvb/gcc-4.7-release/x86_64-w64-mingw32-gcc-4.7.2-release-win64_rubenvb.7z/download">Download and Unarchive MinGW-w64 GCC 4.7.2</a>
 *  -# Use the free open source program <a href="http://www.7-zip.org/">7-Zip</a> to unarchive.
 *  -# Move "x86_64-w64-mingw32-gcc-4.7.2-release-win64_rubenvb\mingw64" to "C:\".
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe">Download and Install CMake 2.8.10.2</a>
 *  -# During installation setup select "add CMake to PATH".
 * -# <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.5/opencv-2.4.5.tar.gz">Download OpenCV 2.4.5</a>
 *  -# Consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a> if you need a program to unarchive tarballs.
 *  -# Move the "opencv-2.4.5" folder to "C:\".
 *  -# From the MinGW-w64 Command Prompt (double-click "C:\mingw64\mingw64env.cmd"):
 *  \code
 *  $ cd C:\opencv-2.4.5
 *  $ mkdir build-mingw64
 *  $ cd build-mingw64
 *  $ cmake -G "MinGW Makefiles" -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=OFF -DCMAKE_BUILD_TYPE=Debug ..
 *  $ mingw32-make
 *  $ mingw32-make install
 *  $ cmake -DCMAKE_BUILD_TYPE=Release ..
 *  $ mingw32-make
 *  $ mingw32-make install
 *  $ mingw32-make clean
 *  \endcode
 * -# <a href="http://download.qt-project.org/official_releases/qt/5.0/5.0.2/single/qt-everywhere-opensource-src-5.0.2.zip">Download and Unzip Qt 5.0.2</a>
 *  -# Install Perl/Python/Ruby dependencies as explained in the "Windows" section of "README". Make sure they are added to "path" when given the option during installation.
 *  -# <a href="http://www.microsoft.com/en-us/download/confirmation.aspx?id=6812">Download and Install Direct X Software Developement Kit</a>, you may also need to install the latest OpenGL drivers from your graphics card manufacturer.
 *  -# From the MinGW-w64 Command Prompt:
 *  \code
 *  $ cd qt-everywhere-opensource-src-5.0.2
 *  $ configure -prefix C:\Qt\Qt5.0.2\5.0.2\mingw64 -opensource -confirm-license -nomake examples -nomake tests -opengl desktop
 *  $ mingw32-make
 *  $ mingw32-make install
 *  $ cd ..
 *  $ rmdir /Q /S qt-everywhere-opensource-src-5.0.2
 *  \endcode
 *  -# mingw32-make will take several hours to finish.
 * -# Create a <a href="github.com">GitHub</a> account and follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
 *  -# Launch "Git Bash" from the Desktop and clone OpenBR:
 *  \code
 *  $ cd /c
 *  $ git clone https://github.com/biometrics/openbr.git
 *  $ cd openbr
 *  $ git submodule init
 *  $ git submodule update
 *  \endcode
 * -# Build OpenBR!
 *  -# From the MinGW-w64 Command Prompt:
 *  \code
 *  $ cd C:\openbr
 *  $ mkdir build-mingw64
 *  $ cd build-mingw64
 *  $ cmake -G "CodeBlocks - MinGW Makefiles" -DCMAKE_RC_COMPILER="C:/mingw64/bin/windres.exe" -DCMAKE_PREFIX_PATH="C:/opencv-2.4.5/build-mingw64/install;C:/Qt/Qt5.0.2/5.0.2/mingw64" -DCMAKE_INSTALL_PREFIX="./install" -DBR_INSTALL_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release ..
 *  $ mingw32-make
 *  $ mingw32-make install
 *  \endcode
 *  -# Check out the "install" folder.
 * -# Hack OpenBR!
 *  -# From the MinGW-w64 Command Prompt:
 *  \code
 *  $ C:\Qt\Qt5.0.2\Tools\QtCreator\bin\qtcreator.exe
 *  \endcode
 *  -# From the Qt Creator "Tools" menu select "Options..."
 *  -# Under "Kits" select "Desktop (default)"
 *  -# For "Compiler:" select "MinGW (x86 64bit in C:\mingw64\bin)" and click "OK"
 *  -# From the Qt Creator "File" menu select "Open File or Project...".
 *  -# Select "C:\openbr\CMakeLists.txt" then "Open".
 *  -# If prompted for the location of CMake, enter "C:\Program Files (x86)\CMake 2.8\bin\cmake.exe".
 *  -# Browse to your pre-existing build directory "C:\openbr\build-mingw64" then select "Next".
 *  -# Clear any text in the "arguments" box then select "Run CMake" then "Finish".
 *  -# You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.
 * -# (Optional) Package OpenBR!
 *  -# From the MinGW-w64 Command Prompt:
 *  \code
 *  $ cd C:\openbr\build-mingw64
 *  $ cpack -G ZIP
 *  \endcode
 */

/*!
 * \page osx_clang OS X Mountain Lion - Clang/LLVM 3.1 - x64
 * -# Download and install the latest "Xcode" and "Command Line Tools" from the <a href="https://developer.apple.com/downloads/index.action#">Apple Developer Downloads</a> page.
 * -# <a href="http://www.cmake.org/files/v2.8/cmake-2.8.10.2.tar.gz">Download CMake 2.8.10.2</a>
 * \code
 * $ cd ~/Downloads
 * $ tar -xf cmake-2.8.10.2.tar.gz
 * $ cd cmake-2.8.10.2
 * $ ./configure
 * $ make -j4
 * $ sudo make install
 * $ cd ..
 * $ rm -rf cmake-2.8.10.2*
 * \endcode
 * -# <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.5/opencv-2.4.5.tar.gz">Download OpenCV 2.4.5</a>
 * \code
 * $ cd ~/Downloads
 * $ tar -xf opencv-2.4.5.tar.gz
 * $ cd opencv-2.4.5
 * $ mkdir build
 * $ cd build
 * $ cmake -DCMAKE_BUILD_TYPE=Release ..
 * $ make -j4
 * $ sudo make install
 * $ cd ../..
 * $ rm -rf opencv-2.4.5*
 * \endcode
 * -# <a href="http://download.qt-project.org/official_releases/qt/5.0/5.0.2/qt-mac-opensource-5.0.2-clang-offline.dmg">Download and install Qt 5.0.2</a>
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * $ cd openbr
 * $ git submodule init
 * $ git submodule update
 * \endcode
 * -# Build OpenBR!
 * \code
 * $ mkdir build # from the OpenBR root directory
 * $ cd build
 * $ cmake -DCMAKE_PREFIX_PATH=~/Qt5.0.2/5.0.2/clang_64 -DCMAKE_BUILD_TYPE=Release ..
 * $ make -j4
 * $ sudo make install
 * \endcode
 * -# Hack OpenBR!
 *  -# Open Qt Creator IDE
 *  \code
 *  $ open ~/Qt5.0.2/Qt\ Creator.app
 *  \endcode
 *  -# From the Qt Creator "File" menu select "Open File or Project...".
 *  -# Select "openbr/CMakeLists.txt" then "Open".
 *  -# Browse to your pre-existing build directory "openbr/build" then select "Continue".
 *  -# Select "Run CMake" then "Done".
 *  -# You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.
 * -# (Optional) Test OpenBR!
 * \code
 * $ cd openbr/scripts
 * $ ./downloadDatasets.sh
 * $ cd ../build
 * $ make test
 * \endcode
 * -# (Optional) Package OpenBR!
 * \code
 * $ cd openbr/build
 * $ sudo cpack -G TGZ
 * \endcode
 * -# (Optional) Build OpenBR documentation!
 *  -# <a href="ftp://ftp.stack.nl/pub/users/dimitri/doxygen-1.8.2.src.tar.gz">Download Doxygen 1.8.2</a>
 *  \code
 *  $ cd ~/Downloads
 *  $ tar -xf doxygen-1.8.2.src.tar.gz
 *  $ cd doxygen-1.8.2
 *  $ ./configure
 *  $ make -j4
 *  $ sudo make install
 *  $ cd ..
 *  $ rm -rf doxygen-1.8.2*
 *  \endcode
 *  -# Modify build settings and recompile.
 *  \code
 *  $ cd openbr/build
 *  $ cmake -DBR_BUILD_DOCUMENTATION=ON ..
 *  $ make -j4
 *  $ open html/index.html
 *  \endcode
 */

/*!
 * \page linux_all Ubuntu 13.04 - GCC 4.7.3 or ICC 13.1.1 - x64
 * -# Install GCC 4.7.3
 * \code
 * $ sudo apt-get update
 * $ sudo apt-get install build-essential
 * \endcode
 *  -# (Optional) Assuming you meet the eligibility requirements and you want to use ICC instead of GCC, <a href="http://software.intel.com/en-us/non-commercial-software-development">Download and Install Intel C++ Studio XE 2013</a>.
 * -# Install CMake 2.8.10.1
 * \code
 * $ sudo apt-get install cmake cmake-curses-gui
 * \endcode
 * -# <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.5/opencv-2.4.5.tar.gz">Download OpenCV 2.4.5</a>
 * \code
 * $ cd ~/Downloads
 * $ tar -xf opencv-2.4.5.tar.gz
 * $ cd opencv-2.4.5
 * $ mkdir build
 * $ cd build
 * $ cmake -DCMAKE_BUILD_TYPE=Release ..
 * $ make -j4
 * $ sudo make install
 * $ cd ../..
 * $ rm -rf opencv-2.4.5*
 * \endcode
 * -# Install Qt 5.0.1
 * \code
 * $ sudo apt-get install qt5-default libqt5svg5-dev qtcreator
 * \endcode
 * -# Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
 * \code
 * $ git clone https://github.com/biometrics/openbr.git
 * $ cd openbr
 * $ git submodule init
 * $ git submodule update
 * \endcode
 * -# Build OpenBR!
 * \code
 * $ mkdir build # from the OpenBR root directory
 * $ cd build
 * $ cmake -DCMAKE_BUILD_TYPE=Release .. # GCC Only
 * $ cmake -DCMAKE_C_COMPILER=/opt/intel/bin/icc -DCMAKE_CXX_COMPILER=/opt/intel/bin/icpc -DCMAKE_BUILD_TYPE=Release .. # ICC Only
 * $ make -j4
 * $ sudo make install
 * \endcode
 * -# Hack OpenBR!
 *  -# Open Qt Creator IDE
 *  \code
 *  $ qtcreator &
 *  \endcode
 *  -# From the Qt Creator "File" menu select "Open File or Project...".
 *  -# Select "openbr/CMakeLists.txt" then "Open".
 *  -# Browse to your pre-existing build directory "openbr/build" then select "Next".
 *  -# Select "Run CMake" then "Finish".
 *  -# You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.
 * -# (Optional) Test OpenBR!
 * \code
 * $ cd openbr/scripts
 * $ ./downloadDatasets.sh
 * $ cd ../build
 * $ make test
 * \endcode
 * -# (Optional) Package OpenBR!
 * \code
 * $ cd openbr/build
 * $ sudo cpack -G TGZ
 * \endcode
 * -# (Optional) Build OpenBR documentation!
 * \code
 * $ sudo apt-get install doxygen
 * $ cd openbr/build
 * $ cmake -DBR_BUILD_DOCUMENTATION=ON ..
 * $ make -j4
 * $ sudo apt-get install libgnome2-bin
 * $ gnome-open html/index.html
 * \endcode
 */

/*!
 * \page qmake_integration QMake Integration
 * \brief Add OpenBR to your Qt <tt>.pro</tt> project.
 *
 * After completing the \ref installation instructions, try launching Qt Creator and opening <tt>\<path_to_openbr_installation\>/share/openbr/qmake_tutorial/hello.pro</tt>.
 *
 * Happy hacking!
 */

/*!
 * \page algorithm_grammar Algorithm Grammar
 * \brief How algorithms are constructed from string descriptions.
 *
 * <b>So you've run <tt>scripts/helloWorld.sh</tt> and it generally makes sense, except you have no idea what <tt>'Open+Cvt(Gray)+Cascade(FrontalFace)+ASEFEyes+Affine(128,128,0.33,0.45)+CvtFloat+PCA(0.95):Dist(L2)'</tt> means or how it is executed.</b>
 * Well if this is the case, you've found the right documentation.
 * Let's get started!
 *
 * In OpenBR an <i>algorithm</i> is a technique for enrolling templates associated with a technique for comparing them.
 * Recall that our ultimate goal is to be able to say how similar two face images are (or two fingerprints, irises, etc.).
 * Instead of storing the entire raw image for comparison, it is common practice to store an optimized representation, or <i>template</i>, of the image for the task at hand.
 * The process of generating this optimized representatation is called <i>template enrollment</i> or <i>template generation</i>.
 * Given two templates, <i>template comparison</i> computes the similarity between them, where the higher values indicate more probable matches and the threshold for determining what constitutes an adaquate match is determined operationally.
 * The goal of template generation is to design templates that are small, accurate, and fast to compare.
 * Ok, you probably knew all of this already, let's move on.
 *
 * The only way of creating an algorithm in OpenBR is from a text string that describes it.
 * We call this string the <i>algorithm description</i>.
 * The algorithm description is seperated into two parts by a ':', with the left hand side indicating how to generate templates and the right hand side indicating how to compare them.
 * Some algorithms, like \ref cpp_gender_estimation and \ref cpp_age_estimation are <i>classifiers</i> that don't create templates.
 * In these cases, the colon and the template comparison technique can be omitted from the algorithm description.
 *
 * There are several motivations for mandating that algorithms are defined from these strings, here are the most important:
 * -# It ensures good software development practices by forcibly decoupling the development of each step in an algorithm, facilitating the modification of algorithms and the re-use of individual steps.
 * -# It spares the creation and maintainance of a lot of very similar header files that would otherwise be needed for each step in an algorithm, observe the abscence of headers in <tt>openbr/plugins</tt>.
 * -# It allows for algorithm parameter tuning without recompiling.
 * -# It is completely unambiguous, both the OpenBR interpreter and anyone familiar with the project can understand exactly what your algorithm does just from this description.
 *
 * Let's look at some of the important parts of the code base that make this possible!
 *
 * In <tt>AlgorithmCore::init()</tt> in <tt>openbr/core/core.cpp</tt> you can see the code for splitting the algorithm description at the colon:
 * \snippet openbr/core/core.cpp Parsing the algorithm description
 *
 * Shortly thereafter in this function we <i>make</i> the template generation and comparison methods:
 * \snippet openbr/core/core.cpp Creating the template generation and comparison methods
 * These make calls are defined in the public \ref cpp_plugin_sdk and can also be called from end user code.
 *
 * Below we discuss some of the source code for \ref br::Transform::make in <tt>openbr/openbr_plugin.cpp</tt>.
 * Note, \ref br::Distance::make is similar in spirit and will not be covered.
 *
 * One of the first steps when converting the template enrollment description into a \ref br::Transform is to replace the operators, like '+', with their full form:
 * \snippet openbr/openbr_plugin.cpp Make a pipe
 * A pipe (see \ref br::PipeTransform in <tt>openbr/plugins/meta.cpp</tt>) is the standard way of chaining together multiple steps in series to form more sophisticated algorithms.
 * PipeTransform takes a list of transforms, and <i>projects</i> templates through each transform in order.
 *
 * After operator expansion, the template enrollment description forms a tree, and the transform is constructed from this description starting recurively starting at the root of the tree:
 * \snippet openbr/openbr_plugin.cpp Construct the root transform
 *
 * At this point we reach arguably the most important code in the entire framework, the <i>object factory</i> in <tt>openbr/openbr_plugin.h</tt>.
 * The \ref br::Factory class is responsible for constructing an object from a string:
 * \snippet openbr/openbr_plugin.h Factory make
 *
 * Going back to our original example, a \ref br::PipeTransform will be created with \ref br::OpenTransform, \ref br::CvtTransform, \ref br::CascadeTransform, \ref br::ASEFEyesTransform, \ref br::AffineTransform, \ref br::CvtFloatTransform, and \ref br::PCATransform as its children.
 * If you want all the tedious details about what exactly this algoritm does, then you should read the \ref br::Transform::project function implemented by each of these plugins.
 * The brief explanation is that it <i>reads the image from disk, converts it to grayscale, runs the face detector, runs the eye detector on any detected faces, uses the eye locations to normalize the face for rotation and scale, converts to floating point format, and then embeds it in a PCA subspaced trained on face images</i>.
 * If you are familiar with face recognition, you will likely recognize this as the Eigenfaces \cite turk1991eigenfaces algorithm.
 *
 * As a final note, the Eigenfaces algorithms uses the Euclidean distance (or L2-norm) to compare templates.
 * Since OpenBR expects <i>similarity</i> values when comparing templates, and not <i>distances</i>, \ref br::DistDistance will return <i>-log(distance+1)</i> so that larger values indicate more similarity.
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
