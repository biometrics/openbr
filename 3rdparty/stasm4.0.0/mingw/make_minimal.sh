# make_minimal.sh

# -Wno-long-long        disables the "long long warnings" for the OpenCV headers
# -Wno-unused-parameter allows virtual func defs that don't use all params
# -Wno-unknown-pragmas  allows OpenMP pragmas without complaint

g++ -o minimal.exe\
  -O3 -DMOD_1 -Wall -Wextra -pedantic\
 -Wno-long-long -Wno-unused-parameter -Wno-unknown-pragmas\
 -Wstrict-aliasing\
 -IE:/opencv2.4.0/build/include -I../stasm  -I../apps\
 -LE:/opencv2.4.0/build/x86/mingw/bin\
 -lopencv_core240 -lopencv_highgui240 -lopencv_imgproc240 -lopencv_objdetect240\
  ../apps/minimal.cpp ../stasm/*.cpp ../stasm/MOD_1/*.cpp
