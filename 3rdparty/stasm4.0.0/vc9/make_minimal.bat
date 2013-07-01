@rem make_minimal.bat: build the minimal Stasm demo program

set OPENCV_HOME=E:/OpenCV2.4.0

set OPENCV_N=240

set OPENCV_LIBDIR=%OPENCV_HOME%/build/x86/vc10/lib

cl /EHsc /O2 /W3 /nologo ^
  /I../stasm /I%OPENCV_HOME%/build/include ^
  %OPENCV_LIBDIR%/opencv_core%OPENCV_N%.lib ^
  %OPENCV_LIBDIR%/opencv_highgui%OPENCV_N%.lib ^
  %OPENCV_LIBDIR%/opencv_imgproc%OPENCV_N%.lib ^
  %OPENCV_LIBDIR%/opencv_objdetect%OPENCV_N%.lib ^
  ../apps/minimal.cpp ../stasm/*.cpp ../stasm/MOD_1/*.cpp
