cd C:\opencv\sources
mkdir build-msvc2019
cd build-msvc2019
cmake -G "NMake Makefiles" -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=OFF -DCMAKE_BUILD_TYPE=Debug ..
nmake
nmake install
cmake -DCMAKE_BUILD_TYPE=Release ..
nmake
nmake install
nmake clean
