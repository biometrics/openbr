VERSION:=$(shell grep 'The current version of the library' ../NEWS | \
	sed -e 's/.* //' -e 's/\.$$//')

doc: html/index.html

html/index.html: index.html.in
	if test -d html; then rm -rf html/*; else mkdir html; fi
	cp ../LICENSE.txt html/
	sed -e "s%@PROJECT_VERSION@%$(VERSION)%g" \
	index.html.in > html/index.html

PREFIX = /usr/local
DEST = $(PREFIX)/share/doc/RandomLib
DOCDEST = $(DEST)/html
INSTALL = install -b

install: html/index.html
	test -d $(DOCDEST) || mkdir -p $(DOCDEST)
	$(INSTALL) -m 644 html/* $(DOCDEST)/

maintainer-clean:
	rm -rf html

.PHONY: doc install clean
