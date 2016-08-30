################################################################################
#
# Makefile to compile and link C programs
#
# Version valid for Linux machines
#
# "make" compiles and links the specified main programs and modules
# using the specified libraries (if any), and produces the executables
# 
# "make clean" removes all files created by "make"
#
# Dependencies on included files are automatically taken care of
#
################################################################################

all: rmxeq mkdep mkxeq
.PHONY: all

# Local directories
SOURCEDIR=src
INCLUDEDIR=inc
RESDIR=res
DATADIR=dat
DESTDIR=.

EIGENDIR=/users/hannahtillim/include/

# Add sourcedir to vpath
VPATH=$(SOURCEDIR)

# Global macros
MACROS = -DRESDIR=\"$(RESDIR)\" -DDATADIR=\"$(DATADIR)\"

# Compiler options
##################

GCC = g++

CFLAGS = -Wall -w -O3 -Wno-unknown-pragmas -I $(INCLUDEDIR) $(MACROS) -I $(EIGENDIR) -std=c++11
LIBS = /usr/local/shared/gsl/1.16-gcc/lib/*.a

INCPATH = $(CFLAGS)

#####################################

# main programs and required modules 

MAIN =  analyse

MODULES = MVA_Analysis Perceptron

############################## do not change ###################################


SHELL=/bin/bash

CC=$(GCC)

PGMS= $(MAIN) $(MODULES)

INCDIRS = $(INCPATH)

OBJECTS = $(addsuffix .o,$(MODULES))

LDFLAGS = $(LIBS)

-include $(addsuffix .d,$(PGMS))


# rule to make dependencies

$(addsuffix .d,$(PGMS)): %.d: %.cc Makefile
	@ $(CC) -MM -ansi $(INCDIRS) $< -o $@


# rule to compile source programs

$(addsuffix .o,$(PGMS)): %.o: %.cc Makefile
	$(CC) $< -c $(CFLAGS) $(INCDIRS) -o $@


# rule to link object files

$(MAIN): %: %.o $(OBJECTS) Makefile
	$(CC) $< $(OBJECTS) $(CFLAGS) $(LDFLAGS) -o $@


# produce executables

mkxeq: $(MAIN)


# remove old executables and old error log file

rmxeq:
	@ -rm -f $(MAIN); \
        echo "delete old executables"		


# make dependencies

mkdep:  $(addsuffix .d,$(PGMS))
	@ echo "generate tables of dependencies"


# clean directory 

clean:
	@ -rm -rf *.d *.o *~ $(MAIN)

.PHONY: clean

################################################################################


