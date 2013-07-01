@rem The following is a basic check that you have gcc on your path
@which gcc | egrep -i "gcc.*.exe" >NUL && goto doit
@echo Setting mingw environment
@set PATH=C:\Rtools\gcc-4.6.3\bin;%PATH%
:doit

\Rtools\bin\make.exe -f Makefile %1 %2 %3
