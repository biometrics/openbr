LIBSTEM = Random
LIBRARY = lib$(LIBSTEM).a

all: $(LIBRARY)

INCLUDEPATH = ../include

MODULES = Random

REQUIREDHEADERS = Random.hpp RandomCanonical.hpp RandomPower2.hpp \
	RandomEngine.hpp RandomAlgorithm.hpp RandomMixer.hpp RandomSeed.hpp \
	RandomType.hpp

OTHERHEADERS = NormalDistribution.hpp ExponentialDistribution.hpp \
	LeadingZeros.hpp ExponentialProb.hpp RandomSelect.hpp \
	ExactExponential.hpp ExactNormal.hpp ExactPower.hpp RandomNumber.hpp

HEADERS = $(REQUIREDHEADERS) $(OTHERHEADERS)
SOURCES = $(addsuffix .cpp,$(MODULES))
OBJECTS = $(addsuffix .o,$(MODULES))

CC = g++ -g
CXXFLAGS = -g -Wall -Wextra -O3 \
	-funroll-loops -finline-functions -fomit-frame-pointer

CPPFLAGS = -I$(INCLUDEPATH) $(DEFINES)
LDFLAGS = $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	$(AR) r $@ $?

VPATH = ../include/RandomLib

INSTALL = install -b
PREFIX = /usr/local

install: $(LIBRARY)
	test -f $(PREFIX)/lib || mkdir -p $(PREFIX)/lib
	$(INSTALL) -m 644 $^ $(PREFIX)/lib

clean:
	rm -f *.o $(LIBRARY)

TAGS: $(HEADERS) $(SOURCES)
	etags $^

HAVE_SSE2 = \
	$(shell grep "flags\b.*\bsse2\b" /proc/cpuinfo 2> /dev/null | \
	tail -1 | wc -l | tr -d ' \t')

HAVE_ALTIVEC = \
	$(shell arch 2> /dev/null | grep ppc | tail -1 | wc -l | tr -d ' \t')

ifeq ($(HAVE_SSE2),1)
  CXXFLAGS += -msse2

# Include
#   #define HAVE_SSE2 1
# in ../include/RandomLib/Config.h

endif

ifeq ($(HAVE_ALTIVEC),1)
  CXXFLAGS += -maltivec

# Include
#   #define HAVE_ALTIVEC 1
# in ../include/RandomLib/Config.h

endif

Random.o: Config.h $(REQUIREDHEADERS)

.PHONY: all install clean
