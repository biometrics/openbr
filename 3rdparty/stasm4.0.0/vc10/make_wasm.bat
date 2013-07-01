@rem make_wasm.bat: build wasm.exe

set OPENCV_HOME=E:/OpenCV2.4.0

set OPENCV_N=240

set OPENCV_LIBDIR=%OPENCV_HOME%/build/x86/vc10/lib

rc /r /nologo ../apps/win/wasm.rc
copy ..\apps\win\wasm.res .

cl /EHsc /O2 /W3 /nologo ^
  /I../stasm /I%OPENCV_HOME%/build/include ^
  %OPENCV_LIBDIR%/opencv_core%OPENCV_N%.lib ^
  %OPENCV_LIBDIR%/opencv_highgui%OPENCV_N%.lib ^
  %OPENCV_LIBDIR%/opencv_imgproc%OPENCV_N%.lib ^
  %OPENCV_LIBDIR%/opencv_objdetect%OPENCV_N%.lib ^
  advapi32.lib comctl32.lib comdlg32.lib gdi32.lib shell32.lib user32.lib ^
  wasm.res ^
  ../apps/win/wasm.cpp ^
  ../apps/appmisc.cpp ^
  ../apps/win/findfile.cpp ^
  ../apps/win/usermsg.cpp ^
  ../apps/win/writewind.cpp ^
  ../stasm/*.cpp ../stasm/MOD_1/*.cpp
