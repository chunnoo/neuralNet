IDIR =inc
CC=g++-8
CFLAGS=-std=c++17 -I$(IDIR) -O3 -isystem/usr/local/include -isystem/usr/local/include/eigen3 -fopenmp -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2

ODIR=obj
LDIR =lib

SDIR=src

LIBS=

MAINFILE=main

_DEPS = neuralNet.hpp mnist.hpp operations.hpp
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = $(MAINFILE).o neuralNet.o mnist.o operations.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_TESTOBJ = test.o matrix.o
TESTOBJ = $(patsubst %,$(ODIR)/%,$(_TESTOBJ))

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< `libpng-config --cflags` $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ `libpng-config --ldflags` $(CFLAGS) $(LIBS)

test: $(TESTOBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
