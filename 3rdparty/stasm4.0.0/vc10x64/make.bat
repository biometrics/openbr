@rem make.bat for Stasm

@rem   How to install the free Microsoft 64 bit C++ compiler for use with with Visual C 2010.
@rem   --------------------------------------------------------------------------------------
@rem
@rem   These instructions were accurate as of May 2012.
@rem
@rem   You need the 64 bit compiler only if you want to do 64 bit builds.  (For
@rem   32 builds, you can use the standard free version of Visual C 2010.)
@rem   If you don't follow these instructions the install will typically fail,
@rem   at least with the free version of Visual C 2010.
@rem
@rem   o Install Visual C 2010
@rem   o Install Visual C 2010 SP1
@rem   o Install Microsoft Windows SDK For Win7 And Net Framework 4
@rem       but with the compiler box unticked
@rem       Available at www.microsoft.com/en-us/download/details.aspx?id=8442
@rem       and choose the x64 ISO (GRMSDKX_EN_DVD.iso)
@rem   o Install Visual C++ 2010 SP1 Compiler Update for the Windows SDK 7.1
@rem       Available at www.microsoft.com/en-us/download/details.aspx?id=4422
@rem

@rem The following sets the environment for the VC10 x64 tool chain, if necessary
@..\tools\which cl | ..\tools\egrep -i "Visual Studio 10.0.VC.bin.amd64.cl.exe" >NUL && goto label_2
@echo Setting VC10 64 bit environment
@cd "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin"
@if %errorlevel% EQU 0 goto label_1
@echo No such directory: C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin
@exit /B 1
:label_1
call SetEnv /Release /x64
@rem We always use the OpenCV 2.4.0 tbb path because it includes tbb_debug.dll (unlike OpenCV 2.3.1)
@set PATH=C:\OpenCV2.4.0\build\x64\vc10\bin;C:\OpenCV2.4.0\build\common\tbb\intel64\vc10;%PATH%
@cd \b\stasm\vc10x64
:label_2

@nmake -nologo CFG=Release MOD=MOD_1 -f ../vc10/makefile %1 %2 %3
@rem @nmake -nologo CFG=Debug MOD=MOD_1 -f ../vc10/makefile %1 %2 %3
@rem @nmake -nologo CFG=ReleaseWithSymbols MOD=MOD_1 -f ../vc10/makefile %1 %2 %3
