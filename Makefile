include config-host.mak

CFLAGS=-Wall -O2 -g
LDFLAGS=-g
LIBS=
DEFINES+=-D_GNU_SOURCE
TOOLS=qemu-mkcow

all: dyngen $(TOOLS) qemu-doc.html qemu.1
	for d in $(TARGET_DIRS); do \
	make -C $$d $@ || exit 1 ; \
        done

qemu-mkcow: qemu-mkcow.o
	$(HOST_CC) -o $@ $^  $(LIBS)

dyngen: dyngen.o
	$(HOST_CC) -o $@ $^  $(LIBS)

%.o: %.c
	$(HOST_CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

clean:
# avoid old build problems by removing potentially incorrect old files
	rm -f config.mak config.h op-i386.h opc-i386.h gen-op-i386.h op-arm.h opc-arm.h gen-op-arm.h 
	rm -f *.o *.a $(TOOLS) dyngen TAGS qemu.pod
	for d in $(TARGET_DIRS); do \
	make -C $$d $@ || exit 1 ; \
        done

distclean: clean
	rm -f config-host.mak config-host.h
	for d in $(TARGET_DIRS); do \
	rm -f $$d/config.h $$d/config.mak || exit 1 ; \
        done

install: all 
	mkdir -p $(prefix)/bin
	install -m 755 -s $(TOOLS) $(prefix)/bin
	mkdir -p $(sharedir)
	install -m 644 pc-bios/bios.bin pc-bios/vgabios.bin $(sharedir)
	mkdir -p $(mandir)/man1
	install qemu.1 $(mandir)/man1
	for d in $(TARGET_DIRS); do \
	make -C $$d $@ || exit 1 ; \
        done

# various test targets
test speed: all
	make -C tests $@

TAGS: 
	etags *.[ch] tests/*.[ch]

# documentation
qemu-doc.html: qemu-doc.texi
	texi2html -monolithic -number $<

qemu.1: qemu-doc.texi
	./texi2pod.pl $< qemu.pod
	pod2man --section=1 --center=" " --release=" " qemu.pod > $@

FILE=qemu-$(shell cat VERSION)

# tar release (use 'make -k tar' on a checkouted tree)
tar:
	rm -rf /tmp/$(FILE)
	cp -r . /tmp/$(FILE)
	( cd /tmp ; tar zcvf ~/$(FILE).tar.gz $(FILE) )
	rm -rf /tmp/$(FILE)

# generate a binary distribution including the test binary environnment 
BINPATH=/usr/local/qemu-i386

tarbin:
	tar zcvf /tmp/qemu-$(VERSION)-i386-glibc21.tar.gz \
                 $(BINPATH)/etc $(BINPATH)/lib $(BINPATH)/bin $(BINPATH)/usr
	tar zcvf /tmp/qemu-$(VERSION)-i386-wine.tar.gz \
                 $(BINPATH)/wine

ifneq ($(wildcard .depend),)
include .depend
endif
