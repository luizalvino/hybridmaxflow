#VT_MPI_INC = "/usr/lib/openmpi/include/"
#VT_MPI_LIB = "/usr/lib/openmpi/lib/"
#THRUST_INC = -I/home/alvino/Dropbox/private/mestrado/orientacao/2014/mpi+cuda/hibrid_cuda/include

CFLAGS = -g -G
#CFLAGS = -pg


compile:
	nvcc $(CFLAGS) -I$(VT_MPI_INC) $(THRUST_INC) -L$(VT_MPI_LIB) -Xptxas -dlcm=ca -arch sm_20 -lmpi -o run-pushrelabel-mpi-cuda max-flow-mpi-cuda.cu

run:
	mpirun -v -np 9 -machinefile mpi-nodes ./run-pushrelabel-mpi-cuda

sync:
	rsync -av ./ luiz_alvino@172.16.27.88:mpi+cuda/hibrid_cuda/
