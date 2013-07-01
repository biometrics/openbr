# Makefile: mingw makefile for stasm

OPENCV_HOME=E:/opencv2.4.0/build
OPENCV_N=240
EXE=.exe

# Note: high_gui is needed for the apps but not for the Stasm library per se.

LIBRARIES=\
 -L$(OPENCV_HOME)/x86/mingw/bin\
 -lopencv_core$(OPENCV_N)\
 -lopencv_highgui$(OPENCV_N)\
 -lopencv_imgproc$(OPENCV_N)\
 -lopencv_objdetect$(OPENCV_N)

# -Wno-long-long        disables the "long long warnings" for the OpenCV headers
# -Wno-unused-parameter allows virtual func defs that don't use all params
# -Wno-unknown-pragmas  allows OpenMP pragmas without complaint

CFLAGS=-O3 -Wall -Wextra -pedantic\
 -Wno-long-long -Wno-unused-parameter -Wno-unknown-pragmas -Wstrict-aliasing

INCLUDES=-I$(OPENCV_HOME)/include -I../stasm  -I../apps

CXX=g++

vpath %.cpp ../stasm ../stasm/MOD_1 ../apps ../apps/swas

SRC=\
 facedet.cpp\
 initasm.cpp\
 asm.cpp\
 classicdesc.cpp\
 convshape.cpp\
 err.cpp\
 eyedet.cpp\
 eyedist.cpp\
 faceroi.cpp\
 hat.cpp\
 hatdesc.cpp\
 landmarks.cpp\
 misc.cpp\
 pinstart.cpp\
 print.cpp\
 shape17.cpp\
 shapehacks.cpp\
 shapemod.cpp\
 startshape.cpp\
 stasm.cpp\
 stasm_lib.cpp

OBJFILES=$(patsubst %.cpp,obj/%.o,$(SRC))

OBJFILES_MINIMAL=\
	$(OBJFILES) $(patsubst %.cpp,obj/%.o,minimal.cpp)

OBJFILES_MINIMAL2=\
	$(OBJFILES) $(patsubst %.cpp,obj/%.o,minimal2.cpp)

OBJFILES_TEST_STASM_LIB=\
	$(OBJFILES) $(patsubst %.cpp,obj/%.o,test_stasm_lib.cpp)

OBJFILES_TEST_STASM_LIB_ERR=\
	$(OBJFILES) $(patsubst %.cpp,obj/%.o,test_stasm_lib_err.cpp)

OBJFILES_STASM_MAIN=\
	$(OBJFILES) $(patsubst %.cpp,obj/%.o,stasm_main.cpp appmisc.cpp)

OBJFILES_SWAS=\
	$(OBJFILES) $(patsubst %.cpp,obj/%.o,swas.cpp \
 appmisc.cpp shapefile.cpp stasm_regex.cpp fitmeas.cpp fm29.cpp)

DEPENDENCY_FILES=\
	$(patsubst %.o,%.d,\
 $(OBJFILES_MINIMAL)\
 $(OBJFILES_MINIMAL2)\
 $(OBJFILES_TEST_STASM_LIB)\
 $(OBJFILES_TEST_STASM_LIB_ERR)\
 $(OBJFILES_STASM_MAIN)\
 $(OBJFILES_SWAS))

all:\
 obj\
 minimal$(EXE)\
 minimal2$(EXE)\
 test_stasm_lib$(EXE)\
 test_stasm_lib_err$(EXE)\
 stasm$(EXE)\
 swas$(EXE)\
 mdiff$(EXE)

# Create the directory "obj" if necessary (obj is the directory for .o and .d files)

obj:
	@mkdir -p obj

minimal$(EXE): $(OBJFILES_MINIMAL)
	$(CXX) -o $@ $(CFLAGS) $(LIBRARIES) $(OBJFILES_MINIMAL)

minimal2$(EXE): $(OBJFILES_MINIMAL2)
	$(CXX) -o $@ $(CFLAGS) $(LIBRARIES) $(OBJFILES_MINIMAL2)

test_stasm_lib$(EXE): $(OBJFILES_TEST_STASM_LIB)
	$(CXX) -o $@ $(CFLAGS) $(LIBRARIES) $(OBJFILES_TEST_STASM_LIB)

test_stasm_lib_err$(EXE): $(OBJFILES_TEST_STASM_LIB_ERR)
	$(CXX) -o $@ $(CFLAGS) $(LIBRARIES) $(OBJFILES_TEST_STASM_LIB_ERR)

stasm$(EXE): $(OBJFILES_STASM_MAIN)
	$(CXX) -o $@ $(CFLAGS) $(LIBRARIES) $(OBJFILES_STASM_MAIN)

swas$(EXE): $(OBJFILES_SWAS)
	$(CXX) -o $@ $(CFLAGS) $(LIBRARIES) $(OBJFILES_SWAS)

# Invoke g++ to create dependency files *.d in the obj directory.  
# Then invoke g++ to create object files *.o in the obj directory.

# TODO obj/%.o: %.cpp Makefile
obj/%.o: %.cpp
	$(CXX) -MF obj/$(notdir $(patsubst %.cpp,./%.d,$<)) $< -MM -MT $@ $(CFLAGS) $(INCLUDES)
	$(CXX) -o $@ -c $< $(CFLAGS) $(DEFINES) $(INCLUDES)

# incorporate the dependency files obj/*.d created above
-include $(DEPENDENCY_FILES)

.PHONY: clean
clean:
	-rm -rf obj 
	-rm -f *.exe

# mdiff is like diff but ignores [times] in square brackets
mdiff$(EXE): ../apps/mdiff/mdiff.cpp
	$(CXX) -o mdiff$(EXE) $(CFLAGS) ../apps/mdiff/mdiff.cpp

# If any test below fails the make will abort with an error message

test_minimal:
	./minimal
	cmp minimal.bmp ../tests/mingw/minimal/minimal.bmp
	@rm minimal.bmp

test_minimal2:
	./minimal2
	cmp minimal2.bmp ../tests/mingw/minimal2/minimal2.bmp
	@rm minimal2.bmp

test_stasm_lib_multi0_minwidth25:
	./test_stasm_lib 0 25 1 ../tests/data/testmulti.jpg
	./mdiff stasm.log ../tests/mingw/stasm_lib/multi0_minwidth25/stasm.log
	@rm stasm.log test_stasm_lib_auto.bmp test_stasm_lib_pinned.bmp

test_stasm_lib_multi1_minwidth25:
	./test_stasm_lib 1 25 1 ../tests/data/testmulti.jpg
	./mdiff stasm.log ../tests/mingw/stasm_lib/multi1_minwidth25/stasm.log
	@rm stasm.log test_stasm_lib_auto.bmp

test_stasm_lib_multi0_minwidth10:
	./test_stasm_lib 0 10 1 ../tests/data/testmulti.jpg
	./mdiff stasm.log ../tests/mingw/stasm_lib/multi0_minwidth10/stasm.log
	@rm stasm.log test_stasm_lib_auto.bmp

test_stasm_lib_multi1_minwidth10:
	./test_stasm_lib 1 10 1 ../tests/data/testmulti.jpg
	./mdiff stasm.log ../tests/mingw/stasm_lib/multi1_minwidth10/stasm.log
	@rm stasm.log test_stasm_lib_auto.bmp

test_stasm_lib_err:
	./test_stasm_lib_err
	./mdiff stasm.log ../tests/mingw/stasm_lib/test_stasm_lib_err.log
	@rm stasm.log

test_stasm_main:
	./stasm ../data/testface.jpg
	./mdiff stasm.log ../tests/mingw/stasm_main/single/test_stasm_main.log
	cmp testface_stasm.bmp ../tests/mingw/stasm_main/single/testface_stasm.bmp
	@rm stasm.log testface_stasm.bmp
	./stasm -m ../tests/data/testmulti.jpg
	./mdiff stasm.log ../tests/mingw/stasm_main/multi/test_stasm_main_multi.log
	@rm stasm.log testmulti_stasm.bmp
	./stasm -d -m -s -c ../tests/data/testmulti.jpg
	./mdiff stasm.log ../tests/mingw/stasm_main/multi/test_stasm_main_multi_small.log
	cmp testmulti_stasm.bmp ../tests/mingw/stasm_main/multi/testmulti_stasm.bmp
	@rm stasm.log testmulti_stasm.bmp
	./stasm -i -n 68 ../data/testface.jpg
	./mdiff stasm.log ../tests/mingw/stasm_main/multi/test_stasm_main_68.log
	@rm stasm.log

test_swas:
	./swas -f ../tests/data/testface.shape
	./mdiff stasm.log ../tests/mingw/swas/swas.log
	./mdiff fit/testface.fit ../tests/mingw/swas/testface.fit
	@rm fit/testface.fit
	cmp testface_ref.bmp ../tests/mingw/swas/testface_ref.bmp
	cmp testface_stasm.bmp ../tests/mingw/swas/testface_stasm.bmp
	@rm stasm.log testface_stasm.bmp

# TODO test_stasm_lib_err not included below, fails under mingw builds

test:\
 mdiff$(EXE)\
 test_stasm_lib_multi0_minwidth25\
 test_stasm_lib_multi1_minwidth25\
 test_stasm_lib_multi0_minwidth10\
 test_stasm_lib_multi1_minwidth10\
 test_stasm_main\
 test_swas

fulltest:\
 test_minimal\
 test_minimal2\
 test
