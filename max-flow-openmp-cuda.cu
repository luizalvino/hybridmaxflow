#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "grafo-cuda.cuh"

#define INICIALIZADO 1

int main(int argc, char *argv[]) {

	int rank, nprocs;
	rank = 0;
	nprocs = 1;

	Grafo *grafo = Grafo::ImportarDoArquivo(argv[1], rank, nprocs);

	printf("Rank:%d | idInicial:%d - idFinal:%d | - Inicialização completa!\n", rank, grafo->idInicial, grafo->idFinal);

	double tempo1 = second();
	grafo->maxFlowInit();
	printf("proc %d | maxFlowInit tempo = %f\n", rank, second() - tempo1);
	GrafoAloc grafo_aloc = grafo->alocaGrafoDevice();
	ExcessType fluxo = grafo_aloc.grafo_d->fluxoTotalDevice(grafo, grafo_aloc.grafo_tmp, rank, nprocs);
	grafo->finalizar();
	return 0;
}