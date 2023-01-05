# Set the compiler and flags here. Note that the OpenMP option must be set.
# Uncomment below to use the GNU compiler.
F90 = gfortran -std=f2008 -fopenmp
F77 = gfortran -std=legacy -fopenmp
# Uncomment below to use the Intel compiler.
#F90 = ifort -std08 -qopenmp
#F77 = ifort -qopenmp
# Set the build flags here
COMP = -c
# List of object files
OBJS = vtmop_libe.o vtmop.o linear_shepard.o shared_modules.o \
       delsparse.o slatec.o qnstop.o lapack.o blas.o

# Build and test the generator and initializer functions for libEnsemble
all : $(OBJS)

# libE interface
vtmop_libe.o : src/vtmop_libe.f90 vtmop.o
	$(F90) $(COMP) src/vtmop_libe.f90 -o vtmop_libe.o


# Main VTMOP library
vtmop.o : src/vtmop.f90 delsparse.o linear_shepard.o qnstop.o
	$(F90) $(COMP) src/vtmop.f90 -o vtmop.o

# delsparse library, used to generate Delaunay graph
delsparse.o : src/delsparse.f90 shared_modules.o
	$(F90) $(COMP) src/delsparse.f90 -o delsparse.o

# linear Shepard's module
linear_shepard.o : src/linear_shepard.f90 shared_modules.o
	$(F90) $(COMP) src/linear_shepard.f90 -o linear_shepard.o

# QNSTOP contains a latin hypercube function
qnstop.o : src/qnstop.f90 shared_modules.o
	$(F90) $(COMP) src/qnstop.f90 -o qnstop.o

# real_precision module
shared_modules.o : src/shared_modules.f90
	$(F90) $(COMP) src/shared_modules.f90 -o shared_modules.o

# Subset of the slatec library, as needed for solving QPs
slatec.o : src/slatec.f
	$(F77) $(COMP) src/slatec.f -o slatec.o

# Subset of LAPACK library, as needed by VTMOP
lapack.o : src/lapack.f
	$(F77) $(COMP) src/lapack.f -o lapack.o

# Subset of BLAS library, as needed by VTMOP
blas.o : src/blas.f
	$(F77) $(COMP) src/blas.f -o blas.o

# Clean command
clean :
	rm -f *.o *.mod *.so
