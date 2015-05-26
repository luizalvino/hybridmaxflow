#include <stdio.h>
#include <stdlib.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <vector>
#include <queue>
#include <algorithm>
#include "fila-cuda.cuh"
#include <thrust/system/cpp/execution_policy.h>
#include <cuda_runtime.h>
#include <omp.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define MIN(x, y) x < y ? x : y
#define MAX(x, y) x > y ? x : y
#define ENQUEUE(v) { \
		ativo[v] = true; \
		filaAtivos->enfileirar(v); \
}
#define ENQUEUE_DEVICE(v) { \
	if (excess[v] > 0) { \
		ativo[v] = true; \
	} \
}
#define BFS_UNROLL_STEP(i) { \
	if (i < tam_adj) { \
		Aresta adj = arestas[i + inicial]; \
		if (dist[adj.to] == numVertices && resCap[adj.reversa] > 0 && !marcado[adj.to]) { \
			dist[adj.to] = novaDistancia; \
			bfsQ->enfileirar(adj.to); \
		} \
	} \
}

__device__ int getRank() {
	return blockIdx.y  * gridDim.x  * blockDim.z * blockDim.y * blockDim.x
        + blockIdx.x  * blockDim.z * blockDim.y * blockDim.x
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x
        + threadIdx.x;
}


typedef unsigned long long ExcessType;
typedef unsigned long CapType;

typedef struct _Aresta {
	int from;
	int to;
	int id;
	int reversa;
	bool local;
	int msgId;

	void init(int from, int to, int id, int reversa, bool local) {
    	this->from = from;
    	this->to = to;
    	this->id = id;
    	this->reversa = reversa;
    	this->local = local;
    	// this->msgId = -1;
	}
} Aresta;

struct ComparaArestaFrom {
	__host__ __device__
	bool operator()(const Aresta &a1, const Aresta &a2) {
		return a1.from < a2.from;
	}
};

struct ComparaArestaFromDist {
	int *dist;

	ComparaArestaFromDist(int *dist) {
		this->dist = dist;
	}

	__host__ __device__
	bool operator()(const Aresta &a1, const Aresta &a2) {
		return a1.from == a2.from ? dist[a1.to] < dist[a2.to] : a1.from < a2.from;
	}
};

typedef struct _Vertice {
	int inicial;
	int final;
} Vertice;

typedef struct _LinhaArquivo {
	int from;
	int to;
	int cap;

	void init(int from, int to, int cap) {
		this->from = from;
		this->to = to;
		this->cap = cap;
	}
} LinhaArquivo;

typedef struct _Mensagem{
	int idAresta;
	ExcessType delta;

	_Mensagem() {
		idAresta = -1;
		delta = 0;
	}

} Mensagem;

typedef struct _Grafo Grafo;

__global__ void pushrelabel_kernel(int start, int stop, Vertice const* __restrict__ vertices, int numVertices, Aresta const* __restrict__ arestas, bool *ativo,
	ExcessType *excess, int *dist, CapType *resCap,	int idInicial, int idFinal, Mensagem *mensagensAnt,	Mensagem *mensagensProx);
__global__ void bfs_step(Grafo *grafo, bool *visitados, bool *ativos, bool *continua);
__global__ void bfs_check(Grafo *grafo, bool *ativos, bool *continua);

typedef struct _GrafoAloc{
	Grafo *grafo_d;
	Grafo *grafo_tmp;
} GrafoAloc;

typedef struct _Grafo {
	Vertice *vertices;
	ExcessType *excess;
	int *dist;
	CapType *resCap;
	int numVertices;
	Aresta *arestas;
	int numArestas;
	bool *ativo;
	ExcessType *excessTotal;
	bool *marcado;
	int idInicial;
	int idFinal;
	int idNum;
	int vertices_por_processo;
	int numVizinhosAnt;
	int numVizinhosProx;
	Mensagem *mensagensAnt;
	Mensagem *mensagensProx;
	std::vector<int> fronteiraAnt;
	std::vector<int> fronteiraProx;

	void init(int _numVertices, int _numArestas, int rank, int nprocs) {
		numVertices = _numVertices;
		numArestas = 0;
		vertices = new Vertice[_numVertices];
		arestas = new Aresta[_numArestas * 2];
		excess = new ExcessType[numVertices];
		resCap = new CapType[_numArestas * 2];
		dist = new int[numVertices];
		ativo = new bool[numVertices];
		marcado = new bool[numVertices];

		int i;
		for (i = 0; i < numVertices; i++) {
			excess[i] = 0;
			dist[i] = 0;
			ativo[i] = false;
			marcado[i] = false;
		}

		excessTotal = new ExcessType;
		*excessTotal = 0;

		vertices_por_processo = ceil(numVertices / nprocs) + 1;
		idInicial = (vertices_por_processo) * rank;
		idFinal = (idInicial + vertices_por_processo) < numVertices ? idInicial + vertices_por_processo - 1: numVertices - 1;
		idNum = rank;
		numVizinhosAnt = 0;
		numVizinhosProx = 0;
		printf("vertices_por_processo = %d\n", vertices_por_processo);
	}

	static struct _Grafo* ImportarDoArquivo(char *arquivo, int rank, int nprocs) {
		struct _Grafo *novoGrafo;
		FILE *file = fopen(arquivo, "r");
		char line[50];
		int n, m, a, b, c;
		if (fgets(line, 50, file) == NULL) {
			printf("Erro!\n");
		}

		if (line[0] == 'c' || line[0] == 'p') {
			do {
				if (line[0] == 'p') {
					sscanf(line, "%*c %*s %d %d", &n, &m);
					novoGrafo = new Grafo;
					novoGrafo->init(n, m, rank, nprocs);
				} else if (line[0] == 'a') {
					sscanf(line, "%*c %d %d %d", &a, &b, &c);
					novoGrafo->AdicionarAresta(a - 1, b - 1, c);
					// linhas[a - 1][numAdjacentes[a - 1]].init(a - 1, b - 1, c);
					// numAdjacentes[a - 1]++;
				}

			} while (fgets(line, 50, file) != NULL);
		} else {
			sscanf(line, " %d %d", &m, &n);
			novoGrafo = new Grafo;
			novoGrafo->init(n, m, rank, nprocs);
			for (int i = 0; i < m; i++) {
				if (fscanf(file, " %d %d %d", &a, &b, &c) == 0) {
					printf("Erro!\n");
				}
				novoGrafo->AdicionarAresta(a, b, c);
			}
		}
		fclose(file);

		thrust::sort(novoGrafo->arestas, novoGrafo->arestas + novoGrafo->numArestas, ComparaArestaFrom());
		int current = -1;
		for (int i = 0; i < novoGrafo->numArestas; ++i) {
			Aresta a = novoGrafo->arestas[i];
			if (a.from != current) {
				current = a.from;
				novoGrafo->vertices[a.from].inicial = i;
				if (current > 0) novoGrafo->vertices[current - 1].final = i;
			}
		}
		novoGrafo->vertices[novoGrafo->numVertices - 1].final = novoGrafo->numArestas;

		novoGrafo->mensagensAnt = new Mensagem[novoGrafo->numVizinhosAnt];
		novoGrafo->mensagensProx = new Mensagem[novoGrafo->numVizinhosProx];

		printf("rank %d | numVizinhosAnt %d | numVizinhosProx %d\n", rank, novoGrafo->numVizinhosAnt, novoGrafo->numVizinhosProx);
		return novoGrafo;
	}

	void AdicionarAresta(int from, int to, int cap) {
		resCap[numArestas] = cap;
		arestas[numArestas].init(from, to, numArestas, numArestas + 1, (from >= idInicial || from <= idFinal));
		numArestas++;
		resCap[numArestas] = 0;
		arestas[numArestas].init(to, from, numArestas, numArestas - 1, (from >= idInicial || from <= idFinal));
		numArestas++;

		if (abs(from - to) > vertices_por_processo) {
			printf("alem da fronteira normal!\n");
		}

		if (to < idInicial && from >= idInicial && from <= idFinal) {
			arestas[numArestas - 2].msgId = numVizinhosAnt;
			numVizinhosAnt++;
			fronteiraAnt.push_back(from);
		} else if (to > idFinal && from >= idInicial && from <= idFinal) {
			arestas[numArestas - 2].msgId = numVizinhosProx;
			numVizinhosProx++;
			fronteiraProx.push_back(from);
		}
		if (from < idInicial && to >= idInicial && to <= idFinal) {
			arestas[numArestas - 1].msgId = numVizinhosAnt;
			numVizinhosAnt++;
			fronteiraAnt.push_back(to);
		} else if (from > idFinal && to >= idInicial && from <= idFinal) {
			arestas[numArestas - 1].msgId = numVizinhosProx;
			numVizinhosProx++;
			fronteiraProx.push_back(to);
		}
}

	__host__ void Discharge(int v, Fila *filaAtivos) {
		do {
			int i;
			for (i = vertices[v].inicial; excess[v] > 0 && i < vertices[v].final; i++) {
				Push(arestas + i, filaAtivos);
				if (excess[v] == 0) break;
			}
			if (i >= vertices[v].final) {
				if (dist[v] == numVertices){break;}
				int minD = numVertices;

				for (i = vertices[v].inicial; i < vertices[v].final; i++) {
					if (i < vertices[v].final) {
						Aresta adj = arestas[i];
						if (resCap[adj.id] > 0 && adj.from == v) {
							int d_tmp = dist[adj.to];
							d_tmp++;
							if (d_tmp < minD) {
								minD = d_tmp;
							}
						}
					}
				}

				dist[v] = minD;
				ENQUEUE(v);

				if (dist[v] == numVertices){break;}
			} else {
				break;
			}
		} while (1);
	}

	__host__ void Push(Aresta *a, Fila *filaAtivos) {
		if (resCap[a->id] > 0) {
			ExcessType delta = MIN(excess[a->from], resCap[a->id]);
			if (dist[a->from] > dist[a->to]) {
				resCap[a->id] -= delta;
				resCap[a->reversa] += delta;
				excess[a->to] += delta;
				excess[a->from] -= delta;
				if (!ativo[a->to]) {
					ENQUEUE(a->to);
				}
			}
		}
	}

	__host__ void Relabel(int v, Fila *filaAtivos) {
		int minD = numVertices;

		int i;
		for (i = vertices[v].inicial; i < vertices[v].final; i++) {
			Aresta *adj = arestas + i;
			if (resCap[adj->id] != 0) {
				int d = dist[adj->to];
				d++;
				if (d < minD) {
					minD = d;
				}
			}
		}
		if (minD != numVertices) {
			dist[v] = minD;
		}
		ENQUEUE(v);
	}

	__device__ __host__ ExcessType fluxoTotal() {
		return excess[numVertices - 1];
	}

	__host__ bool bfsDevice(Grafo *grafo_d) {
		bool *visitados = (bool *) calloc(numVertices, sizeof(bool));
		bool *ativos = (bool *) calloc(numVertices, sizeof(bool));
		bool *ativos_d;
		bool *visitados_d;
		bool *continua = new bool;
		bool *continua_d;
		gpuErrchk( cudaMalloc((void **) &visitados_d, sizeof(bool) * numVertices) );
		gpuErrchk( cudaMalloc((void **) &ativos_d, sizeof(bool) * numVertices) );
		gpuErrchk( cudaMalloc((void **) &continua_d, sizeof(bool)) );

		ativos[numVertices - 1] = true;
		visitados[numVertices - 1] = true;
		gpuErrchk( cudaMemcpy(ativos_d, ativos, sizeof(bool) * numVertices, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(visitados_d, visitados, sizeof(bool) * numVertices, cudaMemcpyHostToDevice) );

		dim3 blocos = numVertices % 256 == 0 ? numVertices / 256 : numVertices / 256 + 1;

		*continua = true;
		while(*continua) {
			*continua = false;
			gpuErrchk( cudaMemcpy(continua_d, continua, sizeof(bool), cudaMemcpyHostToDevice) );
			bfs_step<<<blocos, 256>>>(grafo_d, visitados_d, ativos_d, continua_d);
			gpuErrchk( cudaMemcpy(continua, continua_d, sizeof(bool), cudaMemcpyDeviceToHost) );
			// printf("continua = %d\n", *continua);
		}
		gpuErrchk( cudaPeekAtLastError() );

		gpuErrchk( cudaMemcpy(visitados, visitados_d, sizeof(bool) * numVertices, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < numVertices; ++i) {
			if (!visitados[i] && !marcado[i]) {
				marcado[i] = true;
				(*excessTotal) = (*excessTotal) - excess[i];
			}
		}

		return false;
	}

	__host__ int bfs() {
		double time1 = second();
		int *f1 = (int *) malloc(sizeof(int) * numVertices);
		int *f2 = (int *) malloc(sizeof(int) * numVertices);
		int f1_size = 1;
		int f2_size;
		int aSize = 0;
		int numMarcados = 0;
		bool chegouFonte = false;

		thrust::fill(dist + 1, dist + numVertices, numVertices);

		f1[0] = numVertices - 1;
		dist[numVertices - 1] = 0;

		int k = 1;
		while(f1_size > 0) {
			// printf("f1 size %d\n", f1_size);
			f2_size = 0;
			#pragma omp parallel
			{
				int v, i, j, stop;
				Vertice *vertice;
				#pragma omp for nowait
				for (i = 0; i < f1_size; i++) {
					for (v = f1[i], vertice = vertices + v, j = vertice->inicial, stop = vertices[v].final; j < stop; ++j) {
						Aresta adj = arestas[j];
						if (dist[adj.to] == numVertices && resCap[adj.reversa] > 0 && !marcado[adj.to]) {
							dist[adj.to] = k;
							int index;
							#pragma omp atomic capture
							{
								index = f2_size;
								f2_size += 1;
							}
							f2[index] = adj.to;
						}
					}
				}
			}

			int *tmp = f1;
			f1 = f2;
			f2 = tmp;
			f1_size = f2_size;
			k++;
		}

		if (*excessTotal > 0) {
			#pragma omp parallel
			{
				#pragma omp for nowait
				for (int i = 1; i < numVertices - 1; ++i) {
					if (dist[i] == numVertices && !marcado[i]) {
						// printf("ativo[%d] = %d\n", i, ativo[i]);
						numMarcados++;
						marcado[i] = true;
						ativo[i] = false;
						#pragma omp atomic
						(*excessTotal) -= excess[i];
					}
				}
			}
		}

		dist[0] = numVertices;

		printf("tempo bfs = %f\n", second() - time1);
		// printf("numVisitados = %d | aSize = %d | marcados = %d | chegouFonte = %d | e[0] = %llu | e[n-1] = %llu | excessTotal = %llu\n", numVisitados, aSize, numMarcados, chegouFonte, excess[0], excess[numVertices - 1], *excessTotal);

		return (numMarcados == 0 && chegouFonte == 0) || aSize == 0;
	}

	__host__ void maxFlowInit() {
		double tempo1 = second();
		bfs();
		thrust::sort(arestas, arestas + numArestas, ComparaArestaFromDist(dist));
		printf("bfs inicial tempo = %f\n", second() - tempo1);

		Fila *filaAtivos = new Fila;
		filaAtivos->init();

		for (int i = vertices[0].inicial; i < vertices[0].final; ++i) {
			Aresta *adjacente = arestas + i;
			excess[0] += resCap[adjacente->id];
			(*excessTotal) += excess[0];
			Push(adjacente, filaAtivos);
		}

		// int contador = 0;
		// while(filaAtivos->tamanho > 0 && filaAtivos->tamanho < 1500) {
		// 	int v = filaAtivos->desenfileirar();
		// 	if (v == numVertices - 1) continue;
		// 	ativo[v] = false;
		// 	Discharge(v, filaAtivos);
		// 	if (contador++ >= numVertices * 2) {
		// 		// printf("bfs init | tamFila = %d\n", filaAtivos->tamanho);
		// 		bfs();
		// 		if (achouFluxoMaximo()) break;
		// 		contador = 0;
		// 	}
		// }
		printf("tempo maxFlowInit = %f\n", second() - tempo1);
		printf("filaAtivos = %d\n", filaAtivos->tamanho);
	}

	_GrafoAloc alocaGrafoDevice() {
		cudaStream_t cs[10];
		for (int i = 0; i < 10; i++) {
			gpuErrchk( cudaStreamCreate(cs + i) );
		}
		double tempo1 = second();
		Grafo *grafo_tmp;
		Vertice *vertices_tmp;
		Fila *filaAtivos_tmp, *filaProx_tmp;
		gpuErrchk( cudaMallocHost(&grafo_tmp, sizeof(Grafo)) );
		gpuErrchk( cudaMallocHost(&vertices_tmp, sizeof(Vertice) * numVertices) );
		gpuErrchk( cudaMallocHost(&filaAtivos_tmp, sizeof(Fila)) );
		gpuErrchk( cudaMallocHost(&filaProx_tmp, sizeof(Fila)) );
		memcpy(grafo_tmp, this, sizeof(Grafo));
		memcpy(vertices_tmp, vertices, sizeof(Vertice) * numVertices);
		gpuErrchk( cudaMalloc(&grafo_tmp->vertices, sizeof(Vertice) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->arestas, sizeof(Aresta) * numArestas) );
		gpuErrchk( cudaMalloc(&grafo_tmp->excess, sizeof(ExcessType) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->resCap, sizeof(CapType) * numArestas) );
		gpuErrchk( cudaMalloc(&grafo_tmp->dist, sizeof(int) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->ativo, sizeof(bool) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->mensagensAnt, sizeof(Mensagem) * numVizinhosAnt) );
		gpuErrchk( cudaMalloc(&grafo_tmp->mensagensProx, sizeof(Mensagem) * numVizinhosProx) );

		gpuErrchk( cudaMemcpyAsync(grafo_tmp->vertices, vertices_tmp, sizeof(Vertice) * numVertices, cudaMemcpyHostToDevice, cs[2]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->excess, excess, sizeof(ExcessType) * numVertices, cudaMemcpyHostToDevice, cs[3]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->resCap, resCap, sizeof(CapType) * numArestas, cudaMemcpyHostToDevice, cs[4]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->dist, dist, sizeof(int) * numVertices, cudaMemcpyHostToDevice, cs[5]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->arestas, arestas, sizeof(Aresta) * numArestas, cudaMemcpyHostToDevice, cs[6]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->ativo, ativo, sizeof(bool) * numVertices, cudaMemcpyHostToDevice, cs[7]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->mensagensAnt, mensagensAnt, sizeof(Mensagem) * numVizinhosAnt, cudaMemcpyHostToDevice, cs[8]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->mensagensProx, mensagensProx, sizeof(Mensagem) * numVizinhosProx, cudaMemcpyHostToDevice, cs[9]) );

		gpuErrchk( cudaDeviceSynchronize() );
		printf("Tempo alocação = %f\n", second() - tempo1);
		GrafoAloc grafo_aloc;
		grafo_aloc.grafo_tmp = grafo_tmp;
		return grafo_aloc;
	}

	__host__ ExcessType fluxoTotalDevice(Grafo *grafo_h, Grafo *grafo_tmp, int rank, int nproc) {
		bool *continuar;
		int global_enviou;
		ExcessType *fluxoTotal;
		double tempo1 = 0, tempo2 = 0, tempo3 = 0, tempo4 = 0, tempoTotal = 0, tempoMsg = 0, tempoCopia = 0;
		unsigned long long i = 0;
		int num_streams = 4;
		int num_blocos = ceil((double)grafo_h->vertices_por_processo / (256 * num_streams)) / 2;
		printf("num_blocks = %d\n", num_blocos);
		dim3 threads_per_block = 256;
		dim3 blocks = num_blocos;
		int loop_size = 256 * num_blocos;
		int loops = ceil(sqrt(num_blocos*256));
		printf("loops = %d\n", loops);
		// dim3 blocks = 128;

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		double tp1 = second();
		gpuErrchk( cudaHostAlloc((void **)&fluxoTotal, sizeof(ExcessType), cudaHostAllocMapped) );
		gpuErrchk( cudaHostAlloc((void **)&continuar, sizeof(bool), cudaHostAllocMapped) );
		*fluxoTotal = 0;

		printf("tempo inicial = %f\n", second() - tp1);
		cudaStream_t streams[num_streams];
		for (int s = 0; s < num_streams; ++s) {
			gpuErrchk( cudaStreamCreate(&streams[s]) );
		}

		cudaDeviceSynchronize();

		MPI_Barrier(MPI_COMM_WORLD);
		tempoTotal = second();
		do {
			// printf("i:%d\n", i);
			*continuar = false;
			tempo1 = second();
			for (int l = 0; l < 256; ++l) {
				for (int start = grafo_h->idInicial; start <= grafo_h->idFinal; start += loop_size * 8) {
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[0]>>>(start, start + loop_size * (2), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal, grafo_tmp->mensagensAnt, grafo_tmp->mensagensProx);
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[1]>>>(start + loop_size * 2, start + loop_size * (4), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal, grafo_tmp->mensagensAnt, grafo_tmp->mensagensProx);
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[2]>>>(start + loop_size * 4, start + loop_size * (6), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal, grafo_tmp->mensagensAnt, grafo_tmp->mensagensProx);
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[3]>>>(start + loop_size * 6, start + loop_size * (8), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal, grafo_tmp->mensagensAnt, grafo_tmp->mensagensProx);
				}
			}
			//gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			tempo2 += second() - tempo1;

			tempoMsg = second();
			if (i % 1 == 0) {

				tempoCopia = second();
				gpuErrchk( cudaMemcpyAsync(grafo_h->resCap, grafo_tmp->resCap, sizeof(CapType) * grafo_h->numArestas, cudaMemcpyDeviceToHost, streams[0]) );
				gpuErrchk( cudaMemcpyAsync(grafo_h->ativo, grafo_tmp->ativo, sizeof(bool) * grafo_h->numVertices, cudaMemcpyDeviceToHost, streams[1]) );
				// gpuErrchk( cudaMemcpy(grafo_h->dist, grafo_tmp->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyDeviceToHost) );
				gpuErrchk( cudaMemcpyAsync(grafo_h->excess, grafo_tmp->excess, sizeof(ExcessType) * grafo_h->numVertices, cudaMemcpyDeviceToHost, streams[2]) );
				gpuErrchk( cudaDeviceSynchronize() );
				//tempoTotal += second() - tempoCopia;

				if (nproc > 1) {
					/*
						A ideia aqui é copiar as mensagem da gpu e enviar para os processos corretos,
						depois recebe as mensagens, processa e parte para o próximo loop.
					*/
					gpuErrchk( cudaMemcpy(grafo_h->mensagensAnt, grafo_tmp->mensagensAnt, sizeof(Mensagem) * grafo_h->numVizinhosAnt, cudaMemcpyDeviceToHost) );
					gpuErrchk( cudaMemcpy(grafo_h->mensagensProx, grafo_tmp->mensagensProx, sizeof(Mensagem) * grafo_h->numVizinhosProx, cudaMemcpyDeviceToHost) );

					int destinos[2] = {rank - 1, rank + 1};

					bool enviouFluxo = false;
					for (int destIdx = 0; destIdx < 2; destIdx++) {
						int j = destinos[destIdx];
						long tam, tam_rec;
						Mensagem *mensagens;

						if (j < 0 || j >= nproc) continue;

						if (j == rank - 1) {
							tam = grafo_h->numVizinhosAnt;
							mensagens = grafo_h->mensagensAnt;
						} else if (j == rank + 1) {
							tam = grafo_h->numVizinhosProx;
							mensagens = grafo_h->mensagensProx;
						}

						// if (tam > 0) printf("rank:%d | destino %d | tam %ld\n", rank, j, tam);

						int idArestas[tam];
						ExcessType deltas[tam];

						for (int l = 0; l < tam; l++) {
							idArestas[l] = mensagens[l].idAresta;
							deltas[l] = mensagens[l].delta;
							mensagens[l].delta = 0;
						}

						double tempoSend = second();
						// double tempoSendTotal;
						MPI_Sendrecv(&tam, 1, MPI_UNSIGNED_LONG, j, 0, &tam_rec, 1, MPI_UNSIGNED_LONG, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						tempoTotal += second() - tempoSend;

						int buffSendSize = tam * (sizeof(int) + sizeof(ExcessType));
						char buffSend[buffSendSize];
						int buffRecvSize = tam_rec * (sizeof(int) + sizeof(ExcessType));
						char buffRecv[buffRecvSize];
						int position = 0;
						MPI_Pack(idArestas, tam, MPI_INT, buffSend, buffSendSize, &position, MPI_COMM_WORLD);
						MPI_Pack(deltas, tam, MPI_UNSIGNED_LONG_LONG, buffSend, buffSendSize, &position, MPI_COMM_WORLD);

						tempoSend = second();
						MPI_Sendrecv(buffSend, buffSendSize, MPI_PACKED, j, 0, buffRecv, buffRecvSize, MPI_PACKED, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						tempoTotal += second() - tempoSend;

						int idArestasRec[tam_rec];
						ExcessType deltasRec[tam_rec];
						position = 0;

						MPI_Unpack(buffRecv, buffRecvSize, &position, idArestasRec, tam_rec, MPI_INT, MPI_COMM_WORLD);
						MPI_Unpack(buffRecv, buffRecvSize, &position, deltasRec, tam_rec, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);


						ExcessType totalEnviado = 0;
						for (int l = 0; l < tam_rec; l++) {
							ExcessType delta = deltasRec[l];
							// printf("rank %d | delta %llu\n", rank, delta);
							if (delta > 0) {
								totalEnviado += delta;
								Aresta *a = grafo_h->arestas + idArestasRec[l];
								grafo_h->resCap[a->id] -= delta;
								grafo_h->resCap[a->reversa] += delta;
								grafo_h->excess[a->to] += delta;
								grafo_h->excess[a->from] -= delta;
								grafo_h->ativo[a->to] = true;
							}
						}
						// tempoSendTotal = second() - tempoSend;
						// printf("Total enviado de %d para %d = %llu | tempoSend = %f\n", j, rank, totalEnviado, tempoSendTotal);
						enviouFluxo = enviouFluxo || totalEnviado > 0;
					}

					int local_enviou = enviouFluxo ? 1 : 0;
					MPI_Reduce(&local_enviou, &global_enviou, 1, MPI_INT, MPI_SUM, nproc - 1, MPI_COMM_WORLD);
					// if (rank == nproc - 1) {
					// 	printf("global_enviou = %d\n", global_enviou);
					// }

					/*
					Enviar a capacidade residual para o processo nproc-1 para executar a rotulação global
					*/
					int buffSendSize = grafo_h->numArestas * sizeof(CapType) + grafo_h->vertices_por_processo * sizeof(CapType);
					char *buffSend = (char *) malloc(sizeof(char) * buffSendSize);
					if (rank < nproc - 1) {
						int position = 0;
						MPI_Pack(grafo_h->resCap, grafo_h->numArestas, MPI_UNSIGNED_LONG, buffSend, buffSendSize, &position, MPI_COMM_WORLD);
						MPI_Pack(grafo_h->excess + grafo_h->idInicial, grafo_h->vertices_por_processo, MPI_UNSIGNED_LONG, buffSend, buffSendSize, &position, MPI_COMM_WORLD);
						MPI_Gather(buffSend, buffSendSize, MPI_PACKED, NULL, buffSendSize, MPI_PACKED, nproc - 1, MPI_COMM_WORLD);
						MPI_Request request;
						MPI_Irecv(grafo_h->dist, grafo_h->numVertices, MPI_INT, nproc - 1, 0, MPI_COMM_WORLD, &request);
						free(buffSend);
					} else {
						int position = 0;
						int buffRecvSize = grafo_h->numArestas * sizeof(CapType) + grafo_h->vertices_por_processo * sizeof(CapType);
						int buffRecvSizeTotal = buffRecvSize * (nproc - 1);
						char *buffRecv = (char*) malloc(sizeof(char) * buffRecvSizeTotal);
						double tempoSend = second();
						MPI_Gather(MPI_IN_PLACE, 0, MPI_PACKED, buffRecv, buffRecvSize, MPI_PACKED, nproc - 1, MPI_COMM_WORLD);
						free(buffSend);
						tempoTotal += second() - tempoSend;

						for (int j = 0; j < nproc - 1; j++) {
							int idInicial_tmp = (grafo_h->vertices_por_processo) * j;
						 	int idFinal_tmp = idInicial_tmp + grafo_h->vertices_por_processo;
						 	CapType *resCap_tmp = new CapType[grafo_h->numArestas];
							MPI_Unpack(buffRecv, buffRecvSizeTotal, &position, resCap_tmp, grafo_h->numArestas, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
							MPI_Unpack(buffRecv, buffRecvSizeTotal, &position, grafo_h->excess + idInicial_tmp, grafo_h->vertices_por_processo, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
							tempoSend = second();
							/*
								o tempo está meio ruim neste trecho
								é bom tentar otimizar

								uma ideia é utilizar um array binário com as arestas que possuem capacidade residual
							*/
							for (int k = 0; k < grafo_h->numArestas; k++) {
								Aresta *a = grafo_h->arestas + k;
								if (a->from >= idInicial_tmp && a->from <= idFinal_tmp) {
									grafo_h->resCap[k] = resCap_tmp[k];
								}
							}
							tempoTotal += second() - tempoSend;
							delete(resCap_tmp);
						}
						free(buffRecv);
						tempo1 = second();
						gpuErrchk( cudaMemcpy(grafo_h->dist, grafo_tmp->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyDeviceToHost) );
						grafo_h->bfs();
						tempo3 += second() - tempo1;
						tempoSend = second();
						for (int j = 0; j < nproc - 1; j++) {
							MPI_Request request;
							MPI_Isend(grafo_h->dist, grafo_h->numVertices, MPI_INT, j, 0, MPI_COMM_WORLD, &request);
						}
						tempoTotal += second() - tempoSend;
					}
				} else {
					tempo1 = second();
					gpuErrchk( cudaMemcpyAsync(grafo_h->dist, grafo_tmp->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyDeviceToHost, streams[0]) );
					grafo_h->bfs();
					// grafo_h->bfsDevice(this);
					tempo3 += second() - tempo1;
				}

				/*
					Se achou o fluxo máximo, encerra e notifica os outros processos
				*/
				bool sair;
				if (rank == nproc - 1) {
					sair = grafo_h->achouFluxoMaximo() || (nproc > 1 && grafo_h->excess[grafo_h->numVertices-1] > 0 && global_enviou == 0);
					// if (sair) {
					// 	printf("global_enviou final = %d\n", global_enviou);
					// }
					for (int j = 0; j < nproc - 1; j++) {
						MPI_Request request;
						MPI_Isend(&sair, 1, MPI_CHAR, j, 0, MPI_COMM_WORLD, &request);
					}
				} else {
					MPI_Request request;
					MPI_Status status;
					MPI_Irecv(&sair, 1, MPI_CHAR, nproc - 1, 0, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					if (sair) {
						printf("rank %d | tempo pushrelabel_kernel %f\n", rank, tempo2);
						MPI_Finalize();
						exit(0);
					}
				}
				if (sair) {
					// grafo_h->bfs(&filaBfs);
					break;
				}

				tempoCopia = second();
				if (nproc > 1) {
					gpuErrchk( cudaMemcpyAsync(grafo_tmp->mensagensAnt, grafo_h->mensagensAnt, sizeof(Mensagem) * grafo_h->numVizinhosAnt, cudaMemcpyHostToDevice, streams[0]) ) ;
					gpuErrchk( cudaMemcpyAsync(grafo_tmp->mensagensProx, grafo_h->mensagensProx, sizeof(Mensagem) * grafo_h->numVizinhosProx, cudaMemcpyHostToDevice, streams[1]) );
					gpuErrchk( cudaMemcpyAsync(grafo_tmp->resCap, grafo_h->resCap, sizeof(CapType) * grafo_h->numArestas, cudaMemcpyHostToDevice, streams[2]) );
				}

				gpuErrchk( cudaMemcpyAsync(grafo_tmp->dist, grafo_h->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[0]) );
				gpuErrchk( cudaMemcpyAsync(grafo_tmp->ativo, grafo_h->ativo, sizeof(bool) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[1]) );
				//gpuErrchk( cudaMemcpyAsync(grafo_tmp->excess, grafo_h->excess, sizeof(ExcessType) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[2]) );
				gpuErrchk( cudaDeviceSynchronize() );
				//tempoTotal += second() - tempoCopia;
			}
			tempo4 += second() - tempoMsg;

			i++;
		} while (1);
		*fluxoTotal = *grafo_h->excessTotal;
		tempoTotal = second() - tempoTotal;
		printf("tempo pushrelabel_kernel: %f  i:%llu\n", tempo2, i);
		printf("tempo bfs: %f\n", tempo3);
		printf("tempo msgs: %f\n", tempo4);
		printf("excessTotal %llu\n", *fluxoTotal);
		printf("tempoTotal: %f\n", tempoTotal);
		return *fluxoTotal;
	}


	bool achouFluxoMaximo() {
		// for (int i = 0; i < numVertices; i++) {
		// 	if (excess[i] > 0) {
		// 		printf("rank %d | v %d | dist %d | excess %llu | ativo %d\n", idNum, i, dist[i], excess[i], ativo[i]);
		// 	}
		// }
		printf("excess[%d] = %llu | excessTotal = %llu\n", numVertices - 1, excess[numVertices - 1], *excessTotal);
		return excess[0] + excess[numVertices - 1] >= *excessTotal;
	}

	void finalizar() {
		// cudaFree(this);
		cudaDeviceReset();
	}

} Grafo;

#define VERTICES_POR_THREAD 2

__global__ void pushrelabel_kernel(int start, int stop, Vertice const* __restrict__ vertices, int numVertices, Aresta const* __restrict__ arestas, bool *ativo,
	ExcessType *excess, int *dist, CapType *resCap,	int idInicial, int idFinal, Mensagem *mensagensAnt,	Mensagem *mensagensProx) {
	const int rankBase = getRank() * VERTICES_POR_THREAD + start;

	#pragma unroll
	for (int t = 0; t < VERTICES_POR_THREAD; ++t) {
		int rank = rankBase + t;
		if (rank < stop && rank <= idFinal) {
			if (ativo[rank] && excess[rank] > 0) {
				ativo[rank] = false;
				int v = rank;
				int jD = dist[v];
				int current = vertices[v].inicial;
				int stop = vertices[v].final;
				while(excess[v] > 0 && jD < numVertices) {
					int minD = numVertices;
					#pragma unroll 2
					for (int i = current; i < stop; ++i) {
						Aresta a = arestas[i];
						if (resCap[a.id] > 0) {
							int local_d = dist[a.to];
							if (local_d < minD) {
								minD = local_d;
								current = i;
							}
							ExcessType delta = MIN(excess[v], resCap[a.id]);
							if (delta > 0) {
								if (jD > local_d) {
									atomicAdd((unsigned long long *) &resCap[a.id], -delta);
									atomicAdd((unsigned long long *) &excess[v], -delta);
									atomicAdd((unsigned long long *) &resCap[a.reversa], delta);
									if (a.to < idInicial) {
										mensagensAnt[a.msgId].idAresta = a.id;
										mensagensAnt[a.msgId].delta += delta;
									} else if (a.to > idFinal) {
										mensagensProx[a.msgId].idAresta = a.id;
										mensagensProx[a.msgId].delta += delta;
									} else {
										atomicAdd((unsigned long long *) &excess[a.to], delta);
										//excess[a.to] += delta;
									}
									if (!ativo[a.to] && a.to != numVertices - 1) {
										ENQUEUE_DEVICE(a.to);
									}
								}
							}
							if (excess[v] == 0){break;}
						}
					}

					if (excess[v] > 0 & minD < numVertices) {
						jD = minD + 1;
						dist[v] = jD;
						ENQUEUE_DEVICE(v);
						if (minD == numVertices){break;}
					} else {
						break;
					}
				}
			}
		}
	}
}

__global__ void bfs_step(Grafo *grafo, bool *visitados, bool *ativos, bool *continua) {
	int rank = getRank();
	int loop = 20;

	while (loop-- > 0) {
		if (rank < grafo->numVertices && ativos[rank]) {
			ativos[rank] = false;
			visitados[rank] = true;
			for (int i = grafo->vertices[rank].inicial; i < grafo->vertices[rank].final; ++i) {
				Aresta adj = grafo->arestas[i];
				if (grafo->resCap[adj.reversa] > 0 && !visitados[adj.to]) {
					grafo->dist[adj.to] = grafo->dist[rank] + 1;
					ativos[adj.to] = true;
					visitados[adj.to] = true;
					*continua = true;
				}
			}
		}
	}
}

__global__ void bfs_check(Grafo *grafo, bool *ativos, bool *continua) {
	int rank = getRank();
	if (rank < grafo->numVertices && ativos[rank]) {
		*continua = true;
	}
}