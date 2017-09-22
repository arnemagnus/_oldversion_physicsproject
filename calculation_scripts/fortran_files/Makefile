#!bin/bash

all: fortran f2py

fortran: 
	gfortran -O0 -c numerical_integrators.f90 test.f90
	gfortran -O0 -o fortran_step main.f90 test.f90 numerical_integrators.f90
f2py:
	f2py -m numerical_fortran_integrators -c numerical_integrators.f90
clean:
	rm fortran_step test.mod test.o numerical_integrators.o common_parameters.mod single_step.mod adaptive_step.mod numerical_fortran_integrators.so



