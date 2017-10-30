CFLAGS = -O4 --default-stream per-thread -Xcompiler -fopenmp

NVCC = nvcc
HOST = ubuntu@ec2-52-8-187-29.us-west-1.compute.amazonaws.com

compile:
	$(NVCC) $(CFLAGS) $(THRUST_INC) -Xptxas -v -arch sm_21 -o run-pushrelabel-openmp-cuda max-flow-openmp-cuda.cu

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
