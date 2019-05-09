#!/bin/bash

#gcc gmmnew.c -lgsl -lgslcblas -lm -o gmmnew
mpicc -c parallel.c -o parallel.o
mpicc gmmnew.c -lgsl -lgslcblas -lm -o gmmnew
./gmmnew
python graphify.py