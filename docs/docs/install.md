A hacker's guide to building, editing, and running OpenBR.

---

# Linux

1. Install GCC 4.9.2

        $ sudo apt-get update
        $ sudo apt-get install build-essential

2. Install CMake 3.0.2

        $ sudo apt-get install cmake cmake-curses-gui

3. [Download OpenCV 2.4.11](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.11/opencv-2.4.11.zip/download), **note** [Build OpenCV with video support](https://github.com/biometrics/openbr/wiki/Build-OpenCV-with-Video-Support-on-Ubuntu)

        $ cd ~/Downloads
        $ unzip opencv-2.4.11.zip
        $ cd opencv-2.4.11
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install
        $ cd ../..
        $ rm -rf opencv-2.4.11*


4. Install Qt 5.4.1

        $ sudo apt-get install qt5-default libqt5svg5-dev qtcreator

5. Create a [GitHub](https://github.com/) account, follow their instructions for [setting up Git](https://help.github.com/articles/set-up-git).

        $ git clone https://github.com/biometrics/openbr.git
        $ cd openbr
        $ git checkout v1.1.0
        $ git submodule init
        $ git submodule update

6. Build OpenBR!

        $ mkdir build # from the OpenBR root directory
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install

7. Hack OpenBR!
    1. Open Qt Creator IDE

        $ qtcreator &

    2. From the Qt Creator "File" menu select "Open File or Project...".
    3. Select "openbr/CMakeLists.txt" then "Open".
    4. Browse to your pre-existing build directory "openbr/build" then select "Next".
    5. Select "Run CMake" then "Finish".
    6. You're all set! You can find more information on Qt Creator [here](http://qt-project.org/doc/qtcreator) if you need it.

8. (Optional) Test OpenBR!

        $ cd openbr/scripts
        $ ./downloadDatasets.sh
        $ cd ../build
        $ make test

9. (Optional) Package OpenBR!

        $ cd openbr/build
        $ sudo cpack -G TGZ

10. (Optional) Build OpenBR documentation!
    1. Build the docs

            $ pip install mkdocs
            $ cd openbr/docs
            $ sh build_docs.sh
            $ mkdocs serve

    2. Navigate to `http://127.0.0.1:8000` in your browser to view the docs.

---

# OSX

1. Download and install the latest "XCode" and "Command Line Tools" from the [Apple Developer Downloads](https://developer.apple.com/downloads/index.action#) page.

2. [Download CMake 3.0.2](http://www.cmake.org/files/v3.0/cmake-3.0.2.tar.gz)

            $ cd ~/Downloads
            $ tar -xf cmake-3.0.2.tar.gz
            $ cd cmake-3.0.2
            $ ./configure
            $ make -j4
            $ sudo make install
            $ cd ..
            $ rm -rf cmake-3.0.2*

3. [Download OpenCV 2.4.11](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.11/opencv-2.4.11.zip/download)

        $ cd ~/Downloads
        $ unzip opencv-2.4.11.zip
        $ cd opencv-2.4.11
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install
        $ cd ../..
        $ rm -rf opencv-2.4.11*

4. [Download and install Qt 5.4.1](http://download.qt.io/official_releases/qt/5.4/5.4.1/qt-opensource-mac-x64-clang-5.4.1.dmg)

5. Create a [GitHub](https://github.com/) account, follow their instructions for [setting up Git](https://help.github.com/articles/set-up-git).

        $ git clone https://github.com/biometrics/openbr.git
        $ cd openbr
        $ git checkout v1.1.0
        $ git submodule init
        $ git submodule update

6. Build OpenBR!

        $ mkdir build # from the OpenBR root directory
        $ cd build
        $ cmake -DCMAKE_PREFIX_PATH=~/Qt/5.4.1/clang_64 -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install

7. Hack OpenBR!
    1. Open Qt Creator IDE

            $ open ~/Qt/Qt\ Creator.app

    2. From the Qt Creator "File" menu select "Open File or Project...".
    3. Select "openbr/CMakeLists.txt" then "Open".
    4. Browse to your pre-existing build directory "openbr/build" then select "Continue".
    5. Select "Run CMake" then "Done".
    6. You're all set! You can find more information on Qt Creator [here](http://qt-project.org/doc/qtcreator) if you need it.

8. (Optional) Test OpenBR!

        $ cd openbr/scripts
        $ ./downloadDatasets.sh
        $ cd ../build
        $ make test


9. (Optional) Package OpenBR!

        $ cd openbr/build
        $ sudo cpack -G TGZ


10. (Optional) Build OpenBR documentation!
    1. Build the docs

            $ pip install mkdocs
            $ cd openbr/docs
            $ sh build_docs.sh
            $ mkdocs serve

    2. Navigate to `http://127.0.0.1:8000` in your browser to view the docs.

---

# Windows

1. [Download Visual Studio Express 2013 for Windows Desktop](http://go.microsoft.com/?linkid=9832280&clcid=0x409) and install. You will have to register with Microsoft, but it's free.

2. [Download and Install CMake 3.0.2](http://www.cmake.org/files/v3.0/cmake-3.0.2-win32-x86.exe)
    1. During installation setup select "Add CMake to PATH".

3. [Download OpenCV 2.4.11](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.11/opencv-2.4.11.zip/download)
    1. Consider the free open source program [7-Zip](http://www.7-zip.org/) if you need a program to unarchive tarballs.
    2. Move the "opencv-2.4.11" folder to "C:\".
    3. Open "VS2013 x64 Cross Tools Command Prompt" (from the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2013" -> "Visual Studio Tools" -> "VS2013 x64 Cross Tools Command Prompt") and enter:

            $ cd C:\opencv-2.4.11
            $ mkdir build-msvc2013
            $ cd build-msvc2013
            $ cmake -G "NMake Makefiles" -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=OFF -DCMAKE_BUILD_TYPE=Debug ..
            $ nmake
            $ nmake install
            $ cmake -DCMAKE_BUILD_TYPE=Release ..
            $ nmake
            $ nmake install
            $ nmake clean

4. [Download and Install Qt 5.4.1](http://download.qt.io/official_releases/qt/5.4/5.4.1/qt-opensource-windows-x86-msvc2013_64-5.4.1.exe)

5. Create a [GitHub](https://github.com/) account and follow their instructions for [setting up Git](https://help.github.com/articles/set-up-git).
    1. Launch "Git Bash" from the Desktop and clone OpenBR:

            $ cd /c
            $ git clone https://github.com/biometrics/openbr.git
            $ cd openbr
            $ git checkout v1.1.0
            $ git submodule init
            $ git submodule update

6. Build OpenBR!
    1. From the VS2013 x64 Cross Tools Command Prompt:

            $ cd C:\openbr
            $ mkdir build-msvc2013
            $ cd build-msvc2013
            $ cmake -G "CodeBlocks - NMake Makefiles" -DCMAKE_PREFIX_PATH="C:/opencv-2.4.11/build/install;C:/Qt/Qt5.4.1/5.4/msvc2013_64" -DCMAKE_INSTALL_PREFIX="./install" -DBR_INSTALL_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release ..
            $ nmake
            $ nmake install

    2. Check out the "install" folder.

7. Hack OpenBR!
    1. From the VS2013 x64 Cross Tools Command Prompt:
        $ C:\Qt\Qt5.4.1\Tools\QtCreator\bin\qtcreator.exe
    2. From the Qt Creator "Tools" menu select "Options..."
    3. Under "Kits" select "Desktop (default)"
    4. For "Compiler:" select "Microsoft Visual C++ Compiler 11.0 (x86_amd64)" and click "OK"
    5. From the Qt Creator "File" menu select "Open File or Project...".
    6. Select "C:\openbr\CMakeLists.txt" then "Open".
    7. If prompted for the location of CMake, enter "C:\Program Files (x86)\CMake 3.0.2\bin\cmake.exe".
    8. Browse to your pre-existing build directory "C:\openbr\build-msvc2013" then select "Next".
    9. Select "Run CMake" then "Finish".
    10. You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.

8. (Optional) Package OpenBR!
    1. From the VS2013 x64 Cross Tools Command Prompt:
        $ cd C:\openbr\build-msvc2013
        $ cpack -G ZIP

---

# Raspbian

1. Install CMake 2.8.9

        $ sudo apt-get install cmake


2. Download OpenCV 2.4.9

        $ wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip
        $ unzip opencv-2.4.9.zip
        $ cd opencv-2.4.9
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make
        $ sudo make install
        $ cd ../..
        $ rm -rf opencv-2.4.9*

3. Install Qt5
    1. Modify source list

            $ nano /etc/apt/sources.list

        by changing:

            $ deb http://mirrordirector.raspbian.org/raspbian/ wheezy main contrib non-free rpi

        to:

            $ deb http://mirrordirector.raspbian.org/raspbian/ jessie main contrib non-free rpi

4. Update apt-get

        $ sudo apt-get update

5. Install packages

        $ sudo apt-get install qt5-default libqt5svg5-dev

6. Create a [GitHub](https://github.com/) account, follow their instructions for [setting up Git](https://help.github.com/articles/set-up-git).

        $ git clone https://github.com/biometrics/openbr.git
        $ cd openbr
        $ git checkout v1.1.0
        $ git submodule init
        $ git submodule update


7. Build OpenBR!

        $ mkdir build # from the OpenBR root directory
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make
        $ sudo make install

8. (Optional) Test OpenBR!

        $ cd openbr/scripts
        $ ./downloadDatasets.sh
        $ cd ../build
        $ make test
