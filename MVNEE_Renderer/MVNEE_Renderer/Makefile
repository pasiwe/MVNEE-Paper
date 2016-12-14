#include paths to libraries glm and embree
INCLUDES = -I../include 
#library directories
LIBDIRS = -L../embree-2.9.0.x86_64.linux/lib
#library names
LIBS = -l:libtbbmalloc.so.2 -l:libtbb.so.2 -l:libembree.so.2
#compiler flags: specify C++11 standard lib, OpenMp and math.h
CFLAGS = -std=c++0x -std=gnu++11 -fopenmp -lm #-v #-Xlinker --verbose
#linker flags: specify C++11 standard lib, OpenMP and math.h
LDFLAGS = -std=c++0x -std=gnu++11 -fopenmp -lm -Wl,-rpath='../embree-2.9.0.x86_64.linux/lib'

all: volumerenderer

volumerenderer: Main.o StringParser.o VolumeRenderer.o  
	#source ../embree-2.9.0.x86_64.linux/embree-vars.sh
	g++ $(LDFLAGS) Main.o StringParser.o VolumeRenderer.o -o volumerenderer $(LIBDIRS) $(LIBS)

Main.o: Main.cpp
	g++ -c $(CFLAGS) $(INCLUDES) Main.cpp

StringParser.o: StringParser.cpp
	g++ -c $(CFLAGS) $(INCLUDES) StringParser.cpp

VolumeRenderer.o: VolumeRenderer.cpp
	g++ -c $(CFLAGS) $(INCLUDES) VolumeRenderer.cpp

clean:
	rm *.o volumerenderer
