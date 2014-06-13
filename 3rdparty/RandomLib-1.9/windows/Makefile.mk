PROGRAMS = RandomExample RandomSave RandomThread \
	RandomCoverage RandomExact RandomLambda RandomTime

VSPROJECTS = $(addsuffix -vc8.vcproj,Random $(PROGRAMS)) \
	$(addsuffix -vc9.vcproj,Random $(PROGRAMS))

all:
	@:
install:
	@:
clean:
	@:

.PHONY: all install clean
