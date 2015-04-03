A hacker's guide to building, editing, and running OpenBR.

---

# Linux

1. Install GCC 4.7.3

        $ sudo apt-get update
        $ sudo apt-get install build-essential

2. Install CMake 2.8.10.1

        $ sudo apt-get install cmake cmake-curses-gui

3. <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.5/opencv-2.4.5.tar.gz">Download OpenCV 2.4.5</a>, **note** <a href="https://github.com/biometrics/openbr/wiki/Build-OpenCV-with-Video-Support-on-Ubuntu">this</a>

        $ cd ~/Downloads
        $ tar -xf opencv-2.4.5.tar.gz
        $ cd opencv-2.4.5
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install
        $ cd ../..
        $ rm -rf opencv-2.4.5*


4. Install Qt 5.0.1

        $ sudo apt-get install qt5-default libqt5svg5-dev qtcreator

5. Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.

        $ git clone https://github.com/biometrics/openbr.git
        $ cd openbr
        $ git checkout 0.5
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
    6. You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.

8. (Optional) Test OpenBR!

        $ cd openbr/scripts
        $ ./downloadDatasets.sh
        $ cd ../build
        $ make test

9. (Optional) Package OpenBR!

        $ cd openbr/build
        $ sudo cpack -G TGZ

10. (Optional) Build OpenBR documentation!

        $ sudo apt-get install doxygen
        $ cd openbr/build
        $ cmake -DBR_BUILD_DOCUMENTATION=ON ..
        $ make -j4
        $ sudo apt-get install libgnome2-bin
        $ gnome-open html/index.html

---

# OSX

1. Download and install the latest "Xcode" and "Command Line Tools" from the <a href="https://developer.apple.com/downloads/index.action#">Apple Developer Downloads</a> page.
    1. <a href="http://www.cmake.org/files/v2.8/cmake-2.8.11.2.tar.gz">Download CMake 2.8.11.2</a>

            $ cd ~/Downloads
            $ tar -xf cmake-2.8.11.2.tar.gz
            $ cd cmake-2.8.11.2
            $ ./configure
            $ make -j4
            $ sudo make install
            $ cd ..
            $ rm -rf cmake-2.8.11.2*

2. <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.6.1/opencv-2.4.6.1.tar.gz">Download OpenCV 2.4.6.1</a>

        $ cd ~/Downloads
        $ tar -xf opencv-2.4.6.1.tar.gz
        $ cd opencv-2.4.6.1
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install
        $ cd ../..
        $ rm -rf opencv-2.4.6.1*

3. <a href="http://download.qt-project.org/official_releases/qt/5.1/5.1.1/qt-mac-opensource-5.1.1-clang-offline.dmg">Download and install Qt 5.1.1</a>

4. Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.

        $ git clone https://github.com/biometrics/openbr.git
        $ cd openbr
        $ git checkout 0.5
        $ git submodule init
        $ git submodule update

5. Build OpenBR!

        $ mkdir build # from the OpenBR root directory
        $ cd build
        $ cmake -DCMAKE_PREFIX_PATH=~/Qt5.1.1/5.1.1/clang_64 -DCMAKE_BUILD_TYPE=Release ..
        $ make -j4
        $ sudo make install

6. Hack OpenBR!
    1. Open Qt Creator IDE

            $ open ~/Qt5.1.1/Qt\ Creator.app

    2. From the Qt Creator "File" menu select "Open File or Project...".
    3. Select "openbr/CMakeLists.txt" then "Open".
    4. Browse to your pre-existing build directory "openbr/build" then select "Continue".
    5. Select "Run CMake" then "Done".
    6. You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.

7. (Optional) Test OpenBR!

        $ cd openbr/scripts
        $ ./downloadDatasets.sh
        $ cd ../build
        $ make test


8. (Optional) Package OpenBR!

        $ cd openbr/build
        $ sudo cpack -G TGZ


9. (Optional) Build OpenBR documentation!
    1. <a href="ftp://ftp.stack.nl/pub/users/dimitri/doxygen-1.8.5.src.tar.gz">Download Doxygen 1.8.5</a>

            $ cd ~/Downloads
            $ tar -xf doxygen-1.8.5.src.tar.gz
            $ cd doxygen-1.8.5
            $ ./configure
            $ make -j4
            $ sudo make install
            $ cd ..
            $ rm -rf doxygen-1.8.5*


    2. Modify build settings and recompile.

            $ cd openbr/build
            $ cmake -DBR_BUILD_DOCUMENTATION=ON ..
            $ make -j4
            $ open html/index.html

---

# Windows

1. <a href="http://www.microsoft.com/en-us/download/details.aspx?id=34673">Download Visual Studio 2012 Express Edition for Windows Desktop</a> and install.
    1. Consider the free open source program <a href="http://wincdemu.sysprogs.org">WinCDEmu</a> if you need a program to mount ISO images.
    2. You will have to register with Microsoft after installation, but it's free.
    3. Grab any available <a href="http://www.microsoft.com/visualstudio/eng/downloads#d-visual-studio-2012-update">Visual Studio Updates</a>.
    4. Download and install <a href="http://msdn.microsoft.com/en-us/windows/hardware/hh852363.aspx">Windows 8 SDK</a>.

2. <a href="http://www.cmake.org/files/v2.8/cmake-2.8.11.2-win32-x86.exe">Download and Install CMake 2.8.11.2</a>
    1. During installation setup select "add CMake to PATH".

3. <a href="http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.6.1/opencv-2.4.6.1.tar.gz">Download OpenCV 2.4.6.1</a>
    1. Consider the free open source program <a href="http://www.7-zip.org/">7-Zip</a> if you need a program to unarchive tarballs.
    2. Move the "opencv-2.4.6.1" folder to "C:\".
    3. Open "VS2012 x64 Cross Tools Command Prompt" (from the Start Menu, select "All Programs" -> "Microsoft Visual Studio 2012" -> "Visual Studio Tools" -> "VS2012 x64 Cross Tools Command Prompt") and enter:

            $ cd C:\opencv-2.4.6.1
            $ mkdir build-msvc2012
            $ cd build-msvc2012
            $ cmake -G "NMake Makefiles" -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=OFF -DCMAKE_BUILD_TYPE=Debug ..
            $ nmake
            $ nmake install
            $ cmake -DCMAKE_BUILD_TYPE=Release ..
            $ nmake
            $ nmake install
            $ nmake clean

4. <a href="http://download.qt-project.org/official_releases/qt/5.1/5.1.1/qt-windows-opensource-5.1.1-msvc2012-x86_64-offline.exe">Download and Install Qt 5.1.1</a>

5. Create a <a href="github.com">GitHub</a> account and follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.
    1. Launch "Git Bash" from the Desktop and clone OpenBR:

            $ cd /c
            $ git clone https://github.com/biometrics/openbr.git
            $ cd openbr
            $ git checkout 0.5
            $ git submodule init
            $ git submodule update

6. Build OpenBR!
    1. From the VS2012 x64 Cross Tools Command Prompt:

            $ cd C:\openbr
            $ mkdir build-msvc2012
            $ cd build-msvc2012
            $ cmake -G "CodeBlocks - NMake Makefiles" -DCMAKE_PREFIX_PATH="C:/opencv-2.4.6.1/build-msvc2012/install;C:/Qt/Qt5.1.1/5.1.1/msvc2012_64" -DCMAKE_INSTALL_PREFIX="./install" -DBR_INSTALL_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release ..
            $ nmake
            $ nmake install

    2. Check out the "install" folder.

7. Hack OpenBR!
    1. From the VS2012 x64 Cross Tools Command Prompt:
        $ C:\Qt\Qt5.1.1\Tools\QtCreator\bin\qtcreator.exe
    2. From the Qt Creator "Tools" menu select "Options..."
    3. Under "Kits" select "Desktop (default)"
    4. For "Compiler:" select "Microsoft Visual C++ Compiler 11.0 (x86_amd64)" and click "OK"
    5. From the Qt Creator "File" menu select "Open File or Project...".
    6. Select "C:\openbr\CMakeLists.txt" then "Open".
    7. If prompted for the location of CMake, enter "C:\Program Files (x86)\CMake 2.8\bin\cmake.exe".
    8. Browse to your pre-existing build directory "C:\openbr\build-msvc2012" then select "Next".
    9. Select "Run CMake" then "Finish".
    10. You're all set! You can find more information on Qt Creator <a href="http://qt-project.org/doc/qtcreator">here</a> if you need.

8. (Optional) Package OpenBR!
    1. From the VS2012 x64 Cross Tools Command Prompt:
        $ cd C:\openbr\build-msvc2012
        $ cpack -G ZIP

---

# Raspian

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

6. Create a <a href="github.com">GitHub</a> account, follow their instructions for <a href="https://help.github.com/articles/set-up-git">setting up Git</a>.

        $ git clone https://github.com/biometrics/openbr.git
        $ cd openbr
        $ git checkout 0.5
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
