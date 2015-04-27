#VT_MPI_INC = "/usr/lib/openmpi/include/"
#VT_MPI_LIB = "/usr/lib/openmpi/lib/"
#THRUST_INC = -I/home/alvino/Dropbox/private/mestrado/orientacao/2014/mpi+cuda/hibrid_cuda/include

CFLAGS = -O4 -g
#CFLAGS = -pg

NVCC = nvcc

compile:
	$(NVCC) $(CFLAGS) -I$(VT_MPI_INC) $(THRUST_INC) -L$(VT_MPI_LIB) -Xptxas -dlcm=ca -arch sm_20 -lmpi -o run-pushrelabel-mpi-cuda max-flow-mpi-cuda.cu

run:
	mpirun -v -np 9 -machinefile mpi-nodes ./run-pushrelabel-mpi-cuda

sync:
	# cria diretórios
	ssh -i ../luizalvino-key-par-virginia.pem ubuntu@ec2-52-5-60-136.compute-1.amazonaws.com mkdir -p hibrid_cuda/pushrelabel-mpi-cuda-full/
	# sobe dados
	rsync --progress -zrave "ssh -i ../luizalvino-key-par-virginia.pem" ./ ubuntu@ec2-52-5-60-136.compute-1.amazonaws.com:hibrid_cuda/pushrelabel-mpi-cuda-full/
