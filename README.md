**www.openbiometrics.org**

NOTICE: Please install the latest version of CMake and Make sure OpenCV 4.x and MSVC2019 is installed. 

1.	Identify the latest stable release tag such as “v1.10”. 
2.	Download all OpenBR source code and switch to the latest release tag. There are two method to do that:
        (1)	Download from: https://github.com/lanqiming/openbr directly. Choose “Download zip” and decompress it in the disk C               with file named “openbr”.
        (2)	Download the “Git Bash” and run the scripts below:      
        
            cd c:/
            git clone https://github.com/lanqiming/openbr.git
            cd openbr
            git checkout v1.1.0
            git submodule init
            git submodule update
            
 	Then the file name “openbr” can be found in the c disk. If the file cannot be download, please use the first method.
            
            
3.	Check the openbr file. After download, please check the integrity of file, especially the file named “janus” in the “openbr/openbr/janus”. If this file is an empty file, please download from the following link and replace it. 
https://github.com/lanqiming/INVSC-janus

4.	Please download and install Qt in the following link.
https://download.qt.io/official_releases/online_installers/
Remember to when you are installing please choose version above 5.14, which can support VS2019.

5.	Before install openbr, please check the "CMakeList.txt" in the file "openbr/openbr". Please check the code in the 42 line. If it is not  "target_link_libraries(openbr ${BR_THIRDPARTY_LIBS})", please change it to that. 

6.	Make sure the MSVC2019 and OpenCV 4.x is installed, and we can build the OpenBR now. 
Use the following code in the Command Prompt.

    	cd C:\openbr
    	mkdir build-msvc2019
    	cd build-msvc2019
    	cmake -G "CodeBlocks - NMake Makefiles" -DCMAKE_PREFIX_PATH="C:/opencv/build/install;F:Qt/Qt5/5.15.2/msvc2019_64" -DCMAKE_INSTALL_PREFIX="./install" -DBR_INSTALL_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release ..
    	nmake
    	nmake install
    
7.	Hack OpenBR
    1.	Open the “Qt Creator” in the Qt files.
    2.	From the Qt Creator "Tools" menu select "Options..."
    3.	Under "Kits" select "Desktop Qt MSVC2019 64bit"
    4.	For "Compiler:" select "Microsoft Visual C++ Compiler 11.0 (x86_amd64)" and click "OK"
    5.	From the Qt Creator "File" menu select "Open File or Project...".
    6.	Select "C:\openbr\CMakeLists.txt" then "Open".
    7.	If prompted for the location of CMake, enter "C:\Program Files (x86)\CMake 3.0.2\bin\cmake.exe".
    8.	Browse to your pre-existing build directory "C:\openbr\build-msvc2013" then select "Next".
    9.	Select "Run CMake" then "Finish".
    
For the step 6, if you meet error said that cannot find file named
	Qt5Config.cmake
	qt5-config.cmake

Please add this code to your #Global settings

    set (CMAKE_PREFIX_PATH "F:Qt\\Qt5\\5.15.2\\msvc2019_64\\lib\\cmake\\Qt5\\")

The path should change to your own where have the “Qt5Config.cmake” file. 





