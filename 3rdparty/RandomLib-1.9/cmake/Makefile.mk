DEST = $(PREFIX)/share/cmake/RandomLib

INSTALL=install -b

all:
	@:
install:
	test -d $(DEST) || mkdir -p $(DEST)
	$(INSTALL) -m 644 FindRandomLib.cmake $(DEST)

clean:
	@:

.PHONY: all install clean
