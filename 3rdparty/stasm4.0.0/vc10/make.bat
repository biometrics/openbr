@rem make.bat for Stasm

@rem The following is a basic check that you have Visual Studio 10.0 on your path
@..\tools\which cl | ..\tools\egrep -i "Visual Studio 10.0.VC.bin.cl.exe" >NUL && goto doit
@echo Environment is not VC10 32 bit
@exit /B 1
:doit

@nmake -nologo CFG=Release MOD=MOD_1 -f ../vc10/makefile %1 %2 %3
@rem @nmake -nologo CFG=Debug MOD=MOD_1 -f ../vc10/makefile %1 %2 %3
@rem @nmake -nologo CFG=ReleaseWithSymbols MOD=MOD_1 -f ../vc10/makefile %1 %2 %3
