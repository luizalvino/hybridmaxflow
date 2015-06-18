#VT_MPI_INC = "/usr/lib/openmpi/include/"
#VT_MPI_LIB = "/usr/lib/openmpi/lib/"
#THRUST_INC = -I/home/alvino/Dropbox/private/mestrado/orientacao/2014/mpi+cuda/hibrid_cuda/include

# CFLAGS = --default-stream per-thread -O4
#CFLAGS = -pg - Xcompiler -fopenmp
CFLAGS = -O4 --default-stream per-thread -Xcompiler -fopenmp

NVCC = nvcc
HOST = ubuntu@ec2-52-8-217-72.us-west-1.compute.amazonaws.com

compile:
	$(NVCC) $(CFLAGS) -I$(VT_MPI_INC) $(THRUST_INC) -L$(VT_MPI_LIB) -Xptxas -v -arch sm_21 -lmpi -o run-pushrelabel-mpi-cuda max-flow-mpi-cuda.cu

run:
	mpirun -v -np 9 -machinefile mpi-nodes ./run-pushrelabel-mpi-cuda

sync:
	# cria diretórios
	ssh -i ../luizalvino-keypair-california.pem $(HOST) mkdir -p hibrid_cuda/pushrelabel-mpi-cuda-full/
	# sobe dados
	rsync --progress -zrave "ssh -i ../luizalvino-keypair-california.pem" ./ $(HOST):hibrid_cuda/pushrelabel-mpi-cuda-full/
sync-samuel:
	ssh samuel.ext.facom.ufms.br mkdir -p hibrid_cuda/pushrelabel-mpi-cuda-full/
	rsync --progress -zrav ./ samuel.ext.facom.ufms.br:hibrid_cuda/pushrelabel-mpi-cuda-full/
