cd C:\openbr
mkdir build-msvc2019
cd build-msvc2019
cmake -G "CodeBlocks - NMake Makefiles" -DCMAKE_PREFIX_PATH="C:/opencv/build/install;F:Qt/Qt5/5.15.2/msvc2019_64" -DCMAKE_INSTALL_PREFIX="./install" -DBR_INSTALL_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release ..
nmake
nmake install
