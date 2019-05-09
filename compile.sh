#!/bin/bash

CFLAGS="-std=gnu99 -Wall -I/pdc/vol/gsl/2.3/include"
LFLAGS="-L/pdc/vol/gsl/2.3/lib -lm -lgsl -lgslcblas"

#echo "mpicc -c parallel.c -o parallel.o $CFLAGS"
mpicc -c parallel.c -o parallel.o $CFLAGS
#echo "mpicc gmmnew.c -o gmmnew $CFLAGS $LFLAGS"
mpicc gmmnew.c -o gmmnew $CFLAGS $LFLAGS
