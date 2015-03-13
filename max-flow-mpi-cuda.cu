#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "grafo-cuda.cuh"

#define INICIALIZADO 1

int main(int argc, char *argv[]) {

	int rank, nprocs;
	MPI_Status stat;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
	Grafo *grafo = Grafo::ImportarDoArquivo(argv[1], rank, nprocs);
		
	printf("Rank:%d | idInicial:%d - idFinal:%d | - Inicialização completa!\n", rank, grafo->idInicial, grafo->idFinal);

 //    if (rank == 0) {
 //    	ExcessType fluxo;
	// 	MPI_Recv(&fluxo, 1, MPI_UNSIGNED_LONG_LONG, 1, 0, MPI_COMM_WORLD, &stat);
	// 	printf("Fluxo máximo = %lld\n", fluxo);
	// } else {
	double tempo1 = second();
	grafo->maxFlowInit();
	printf("proc %d | maxFlowInit tempo = %f\n", rank, second() - tempo1);
	GrafoAloc grafo_aloc = grafo->alocaGrafoDevice();
	tempo1 = second();
	ExcessType fluxo = grafo_aloc.grafo_d->fluxoTotalDevice(grafo, grafo_aloc.grafo_tmp, rank, nprocs);
	// ExcessType fluxo = grafo->fluxoTotalDevice(grafo, grafo);
	printf("tempo:%f\n", second() - tempo1);
	//MPI_Send(&fluxo, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
	// }

	MPI_Finalize();
	grafo->finalizar();
	return 0;
}