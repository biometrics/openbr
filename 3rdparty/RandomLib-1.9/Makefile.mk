MAKEFILE := $(lastword $(MAKEFILE_LIST))
MAKE := $(MAKE) -f $(MAKEFILE)
SUBDIRS = src examples doc
ALLDIRS = include $(SUBDIRS) windows cmake

all: src examples

$(SUBDIRS):
	$(MAKE) -C $@

examples: src
install: install-headers install-lib install-examples install-cmake install-doc
clean: clean-src clean-examples clean-doc
maintainer-clean: clean maintainer-clean-doc
install-headers:
	$(MAKE) -C include install
install-lib:
	$(MAKE) -C src install
install-examples: src
	$(MAKE) -C examples install
install-cmake:
	$(MAKE) -C cmake install
install-doc: doc
	$(MAKE) -C doc install
clean-src:
	$(MAKE) -C src clean
clean-examples:
	$(MAKE) -C examples clean
clean-doc:
	$(MAKE) -C doc clean
maintainer-clean-doc:
	$(MAKE) -C doc maintainer-clean

.PHONY: all $(SUBDIRS) install \
	install-headers install-lib install-examples install-cmake install-doc \
	clean clean-src clean-examples clean-doc package \
	maintainer-clean maintainer-clean-doc
