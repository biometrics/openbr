PROGRAMS = RandomExample RandomSave RandomThread \
	RandomCoverage RandomExact RandomLambda RandomTime

ifdef HAVE_MPFR
  MPFREXAMPLE = MPFRExample
  LDLIBS += -lmpfr -lgmp
endif

all: $(PROGRAMS) $(MPFREXAMPLE)

LIBSTEM = Random
LIBRARY = lib$(LIBSTEM).a

INCLUDEPATH = ../include
LIBPATH = ../src

# After installation, use these values of INCLUDEPATH and LIBPATH
# INCLUDEPATH = $(PREFIX)/include
# LIBPATH = $(PREFIX)/lib

REQUIREDHEADERS = Random.hpp RandomCanonical.hpp RandomPower2.hpp \
	RandomEngine.hpp RandomAlgorithm.hpp RandomMixer.hpp RandomSeed.hpp \
	RandomType.hpp

CC = g++ -g
CXXFLAGS = -g -Wall -O3 -funroll-loops -finline-functions -fomit-frame-pointer

CPPFLAGS = -I$(INCLUDEPATH) $(DEFINES)
LDLIBS = -L$(LIBPATH) -l$(LIBSTEM)

$(PROGRAMS): $(LIBPATH)/$(LIBRARY)
	$(CC) $(LDFLAGS) -o $@ $@.o $(LDLIBS)

VPATH = ../include/RandomLib

clean:
	rm -f *.o

PREFIX = /usr/local
# After installation, use these values of CPPFLAGS and LDFLAGS

# CPPFLAGS = -I$(PREFIX)/include
# LDFLAGS = -L$(PREFIX)/lib -l$(LIBSTEM)

ifdef RANDOMLIB_DEFAULT_GENERATOR
  CPPFLAGS += -DRANDOMLIB_DEFAULT_GENERATOR=$(RANDOMLIB_DEFAULT_GENERATOR)
endif

ifdef HAVE_BOOST_SERIALIZATION
  CPPFLAGS += -DHAVE_BOOST_SERIALIZATION=1
  LDLIBS += -lboost_serialization-mt
endif

HAVE_OPENMP=1
ifneq ($(HAVE_OPENMP),0)
  CPPFLAGS += -DHAVE_OPENMP=1
  CXXFLAGS += -fopenmp
  LDFLAGS += -fopenmp
endif

RandomExample.o: Config.h $(REQUIREDHEADERS) \
	NormalDistribution.hpp RandomSelect.hpp
RandomExample: RandomExample.o

RandomSave.o: Config.h $(REQUIREDHEADERS)
RandomSave: RandomSave.o

RandomThread.o: Config.h $(REQUIREDHEADERS)
RandomThread: RandomThread.o

RandomTime.o: Config.h $(REQUIREDHEADERS) \
	NormalDistribution.hpp RandomSelect.hpp
RandomTime: RandomTime.o

RandomCoverage.o: Config.h $(REQUIREDHEADERS) \
	NormalDistribution.hpp ExponentialDistribution.hpp RandomSelect.hpp \
	LeadingZeros.hpp ExponentialProb.hpp RandomNumber.hpp \
	ExactExponential.hpp ExactNormal.hpp ExactPower.hpp
RandomCoverage: RandomCoverage.o

RandomExact.o: Config.h $(REQUIREDHEADERS) \
	RandomNumber.hpp ExactExponential.hpp ExactNormal.hpp \
	ExponentialProb.hpp
RandomExact: RandomExact.o

RandomLambda.o: Config.h $(REQUIREDHEADERS) NormalDistribution.hpp
RandomLambda: RandomLambda.o

MPFRExample.o: MPFRRandom.hpp MPFRNormal.hpp
MPFRExample: MPFRRandom.o

# Examples are not installed
install:
