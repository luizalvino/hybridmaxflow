#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

typedef struct _Fila{
	int valores[526904];
	int primeiro;
	unsigned int tamanho;
	int max_queue_size;
	unsigned int processado;

	__host__
	void init() {
		tamanho = 0;
		primeiro = 0;
		max_queue_size = 526904;
		// valores = new int[max_queue_size];
		// cudaHostAlloc((void **)&valores, sizeof(int) * max_queue_size, cudaHostAllocMapped);
		processado = 0;
	}

	void reset() {
		tamanho = 0;
		primeiro = 0;
		max_queue_size = 526904;
		processado = 0;
	}

	__device__ void enfileirarDevice(int valor) {
		int index = atomicAdd(&tamanho, 1);
		enfileirar(index, valor);
	}

	__device__ __host__
	 void enfileirar(int valor) {
		int index = (primeiro + tamanho) % max_queue_size;
		valores[index] = valor;
		tamanho++;

		// printf("primeiro = %d  tamanho = %d\n", primeiro, tamanho);
	}

	__device__ __host__
	 void enfileirar(int index, int valor) {
		int novoIndex = (primeiro + index) % max_queue_size;
		valores[novoIndex] = valor;
		if (tamanho < index + 1) {
			tamanho = index + 1;
		}
	}

	__device__ __host__
	 int desenfileirar() {
		int ret = valores[primeiro];
		tamanho--;
		primeiro++;
		if (primeiro == max_queue_size) {
			primeiro = 0;
		}
		return ret;
	}

	__device__ __host__
	int valor(int index) {
		int novoIndex = (primeiro + index) % max_queue_size;
		return valores[novoIndex];
	}

	__device__
	// void imprimir() {
	// 	int rank = getRank();
	// 	if (rank == 0) {
	// 		printf("novaFila.tamanho = %d\n", tamanho);
	// 		while (tamanho > 0) {
	// 			printf("desenfileirar = %d\n", desenfileirar());
	// 		}
	// 	}
	// }

	__device__ __host__ int vazia() {
		return tamanho == 0;
	}

}Fila;

double second() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__device__ int fila_index;
__device__ Fila novaFila;

// __global__ void init() {
// 	novaFila.init();
// }

// __global__
//  void queue_kernel(Fila *fila) {
// 	int rank = getRank();

// 	if (rank < fila->tamanho) {
// 		int value = fila->valores[rank];
// 		if (value % 3 == 0 && value % 2 == 0 && value % 4 == 0 && value % 5 == 0) {
// 			novaFila.enfileirarDevice(value);
// 		}
// 	}
// }

// __global__
//  void imprimir() {
// 	novaFila.imprimir();
// }