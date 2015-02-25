#include <stdio.h>
#include <stdlib.h>
#include "fila-cuda.cuh"

#define ADJACENTE(v, index) arestas[vertices[v].adjacentes[index]]
#define ADJREVERSO(v, index) (arestas + arestas[vertices[v].adjacentes[index]].reversa)
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
#define MAX_ADJ 10
#define BFS_UNROLL_STEP(i) { \
	if (i < vertice->numAdjacentes) { \
		Aresta *adj = arestas + vertice->adjacentes[i]; \
		if (resCap[adj->reversa] > 0) { \
			if (dist[adj->to] == numVertices && adj->from == v) { \
				dist[adj->to] = novaDistancia; \
				bfsQ->enfileirar(adj->to); \
			} \
		} \
	} \
}
#define RELABEL_DEVICE_UNROLL_STEP(i) { \
	if (i < tam_adj) { \
		Aresta *adj = arestas + vertice->adjacentes[i]; \
		if (resCap[adj->id] > 0 && adj->from == v) { \
			int d = dist[adj->to]; \
			d++; \
			if (d < minD) { \
				minD = d; \
			} \
		} \
	} \
}

#define DISCHARGE_DEVICE_UNROLL_STEP(i) { \
	if (i < tam_adj) { \
		Aresta *a = arestas + vertice->adjacentes[i]; \
		ExcessType local_resCap = resCap[a->id]; \
		if (local_resCap > 0) { \
			ExcessType delta = MIN(excess[v], local_resCap); \
			if (dist[v] > dist[a->to]) { \
				atomicAdd((unsigned long long *) &resCap[a->id], -delta); \
				atomicAdd((unsigned long long *) &resCap[a->reversa], delta); \
				atomicAdd((unsigned long long *) &excess[a->to], delta); \
				atomicAdd((unsigned long long *) &excess[v], -delta); \
				if (!ativo[a->to]) { \
					ENQUEUE_DEVICE(a->to); \
				} \
			} \
		} \
		if (excess[v] == 0){ ativo[v]=false; break;} \
	} \
}

__device__ int getRank() {
	return blockIdx.y  * gridDim.x  * blockDim.z * blockDim.y * blockDim.x
        + blockIdx.x  * blockDim.z * blockDim.y * blockDim.x
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x
        + threadIdx.x + 1;
}


typedef unsigned long long ExcessType;
typedef unsigned long CapType;

typedef struct _Aresta {
	int from;
	int to;
	int id;
	int index;
	int reversa;

	void init(int from, int to, int id, int index, int reversa) {
    	this->from = from;
    	this->to = to;
    	this->id = id;
    	this->index = index;
    	this->reversa = reversa;
	}
} Aresta;

typedef struct _Vertice {
	int label;
	int adjacentes[70];
	int numAdjacentes;

	void init(int _label) {
		label = _label;
		numAdjacentes = 0;
	}

	void AdicionarAdjacente(int index) {
		// if (numAdjacentes == 0) {
		// 	adjacentes = new int;
		// } else {
		// 	adjacentes = (int *) realloc(adjacentes, sizeof(int) * (numAdjacentes + 1));
		// }

		adjacentes[numAdjacentes] = index;
		numAdjacentes++;
	}
} Vertice;

typedef struct _Grafo Grafo;
__global__ void pushrelabel_kernel(struct _Grafo *grafo, ExcessType *fluxoMaximo);
__global__ void proxima_fila(Grafo *grafo);
__global__ void check_bfs(Grafo *grafo, int cycles);
__global__ void check_flow(Grafo *grafo, ExcessType *fluxoMaximo);
__global__ void check_fim(Grafo *grafo, bool *continuar);


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
	// Fila *filaAtivos;
	// Fila *filaProx;
	bool *ativo;
	// int *count;
	int nm;

	void init(int _numVertices, int _numArestas) {
		numVertices = _numVertices;
		numArestas = 0;
		vertices = new Vertice[_numVertices];
		arestas = new Aresta[_numArestas * 2];
		excess = new ExcessType[numVertices];
		resCap = new CapType[_numArestas * 2];
		dist = new int[numVertices];
		ativo = new bool[numVertices];
		// cudaMallocHost(&vertices, sizeof(Vertice) * numVertices);
		// cudaMallocHost(&arestas, sizeof(Aresta) * _numArestas * 2);
		// cudaMallocHost(&excess, sizeof(ExcessType) * numVertices);
		// cudaMallocHost(&resCap, sizeof(CapType) * _numArestas * 2);
		// cudaMallocHost(&dist, sizeof(int) * numVertices);
		// cudaMallocHost(&ativo, sizeof(bool) * numVertices);

		// filaAtivos = new Fila;
		// filaProx = new Fila;
		// count = new int[numVertices * 2];
		// filaProx->init();
		// filaAtivos->init();
		int i;
		for (i = 0; i < numVertices; i++) {
			vertices[i].init(i);
			excess[i] = 0;
			dist[i] = 0;
			ativo[i] = false;
		}
		nm = 6 * numVertices + numArestas;
	}

	static struct _Grafo* ImportarDoArquivo(char *arquivo) {
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
					// cudaHostAlloc((void **)&novoGrafo, sizeof(Grafo), cudaHostAllocMapped);
					novoGrafo = new Grafo;
					novoGrafo->init(n, m);
				} else if (line[0] == 'a') {
					sscanf(line, "%*c %d %d %d", &a, &b, &c);
					novoGrafo->AdicionarAresta(a - 1, b - 1, c);
					//novoGrafo->adicionarAresta(b - 1, a - 1, grafo, arestas, &aresta_index, 0);
				}

			} while (fgets(line, 50, file) != NULL);
		} else {
			sscanf(line, " %d %d", &m, &n);
			// cudaHostAlloc((void **)&novoGrafo, sizeof(Grafo), cudaHostAllocMapped);
			novoGrafo = new Grafo;
			novoGrafo->init(n, m);
			for (int i = 0; i < m; i++) {
				if (fscanf(file, " %d %d %d", &a, &b, &c) == 0) {
					printf("Erro!\n");
				}
				novoGrafo->AdicionarAresta(a, b, c);
				//novoGrafo->adicionarAresta(b, a, grafo, arestas, &aresta_index, 0);
			}
		}
		return novoGrafo;
	}

	void AdicionarAresta(int from, int to, int cap) {
		resCap[numArestas] = cap;
		arestas[numArestas].init(from, to, numArestas, vertices[to].numAdjacentes, numArestas + 1);
		vertices[from].AdicionarAdjacente(numArestas);
		numArestas++;
		resCap[numArestas] = 0;
		arestas[numArestas].init(to, from, numArestas, vertices[from].numAdjacentes - 1, numArestas - 1);
		vertices[to].AdicionarAdjacente(numArestas);
		numArestas++;
}

	__device__ __host__ void ImprimirArestas() {
		printf("grafo.numVertices = %d  grafo.numArestas = %d\n", numVertices, numArestas);
		for (int i = 0; i < numArestas; i++) {
			printf("%d - aresta(id ->%d, from -> %d, to -> %d, resCap -> %lu, index -> %d)\n", i, arestas[i].id, arestas[i].from, arestas[i].to, resCap[i], arestas[i].index);
		}
	}

	__host__ void Discharge(int v, Fila *filaAtivos) {
		do {
			for (int i = 0; excess[v] > 0 && i < vertices[v].numAdjacentes; i++) {
				Push(&ADJACENTE(v, i), filaAtivos);
				if (excess[v] == 0) break;
			}
			if (excess[v] > 0) {
				if (dist[v] == numVertices) break;
				Relabel(v, filaAtivos);
				// if (count[dist[v]] == 1) {
				//  	Gap(dist[v]);
				// }
				if (dist[v] == numVertices) break;
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
				resCap[ADJACENTE(a->to, a->index).id] += delta;
				excess[a->to] += delta;
				excess[a->from] -= delta;
				if (!ativo[a->to]) {
					ENQUEUE(a->to);
				}
			}
		}
	}

	__host__ void Relabel(int v, Fila *filaAtivos) {
		int tam_adj = vertices[v].numAdjacentes;
		int minD = numVertices;

		for (int i = 0; i < tam_adj; i++) {
			Aresta *adj = &ADJACENTE(v, i);
			if (resCap[adj->id] > 0 && adj->from == v) {
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

	__device__ void DischargeDevice(int v) {
		int tam_adj = vertices[v].numAdjacentes;
		Vertice *vertice = vertices + v;
		ExcessType excess_tmp;
		ExcessType resCap_tmp;
		ExcessType local_resCap;
		int dist_tmp;
		int dist_tmp2;
		Aresta a;
		Aresta a_tmp;
		do {
			excess_tmp = excess[v];
			dist_tmp = numVertices;

			for (int i = 0; i < tam_adj; i++) {
				a_tmp = arestas[vertice->adjacentes[i]];
				local_resCap = resCap[a_tmp.id];
				if (local_resCap > 0) {
					dist_tmp2 = dist[a_tmp.to];
					if (dist_tmp2 < dist_tmp) {
						a = a_tmp;
						dist_tmp = dist_tmp2;
						resCap_tmp = local_resCap;
					}
				}
			}
			if (dist[v] > dist_tmp) {
				ExcessType delta = MIN(excess_tmp, resCap_tmp);
				atomicAdd((unsigned long long *) &resCap[a.id], -delta);
				atomicAdd((unsigned long long *) &resCap[a.reversa], delta);
				atomicAdd((unsigned long long *) &excess[a.to], delta);
				atomicAdd((unsigned long long *) &excess[v], -delta);
				if (excess[a.to] > 0) {
					ENQUEUE_DEVICE(a.to);
				}
			} else {
				dist[v] = dist_tmp + 1;
			}

			if (excess[v] <= 0 || dist[v] >= numVertices) break;

			// for (int i = 0; i < tam_adj; i++) {
			// 	if (i < tam_adj) {
			// 		Aresta *a = arestas + vertice->adjacentes[i];
			// 		ExcessType local_resCap = resCap[a->id];
			// 		excess_tmp = excess[v];
			// 		if (local_resCap > 0) {
			// 			ExcessType delta = MIN(excess_tmp, local_resCap);
			// 			if (dist[v] > dist[a->to]) {
			// 				atomicAdd((unsigned long long *) &resCap[a->id], -delta);
			// 				atomicAdd((unsigned long long *) &resCap[a->reversa], delta);
			// 				atomicAdd((unsigned long long *) &excess[a->to], delta);
			// 				atomicAdd((unsigned long long *) &excess[v], -delta);
			// 				if (excess[a->to] > 0) {
			// 					ENQUEUE_DEVICE(a->to);
			// 				}
			// 			}
			// 		}
			// 		if (excess[v] == 0){ break;}
			// 	}
			// }

			// if (excess[v] > 0) {
			// 	if (dist[v] == numVertices) break;
			// 	RelabelDevice(v);
			// 	if (dist[v] == numVertices) break;
			// } else {
			// 	break;
			// }
		} while (1);
	}

	__device__ void PushDevice(Aresta *a) {
		if (resCap[a->id] > 0) {
			ExcessType delta = MIN(excess[a->from], resCap[a->id]);
			if (dist[a->from] - 1 == dist[a->to]) {
				atomicAdd((unsigned long long *) &resCap[a->id], -delta);
				atomicAdd((unsigned long long *) &resCap[ADJACENTE(a->to, a->index).id], delta);
				atomicAdd((unsigned long long *) &excess[a->to], delta);
				atomicAdd((unsigned long long *) &excess[a->from], -delta);
				if (!ativo[a->to]) {
					ENQUEUE_DEVICE(a->to);
				}
			}
		}
	}

	__device__ int RelabelDevice(int v) {
		int tam_adj = vertices[v].numAdjacentes;
		int minD = numVertices;

		Vertice *vertice = vertices + v;
		for (int i = 0; i < tam_adj; i++) {
			if (i < tam_adj) {
				Aresta *adj = arestas + vertice->adjacentes[i];
				if (resCap[adj->id] > 0 && adj->from == v) {
					int d = dist[adj->to];
					d++;
					if (d < minD) {
						minD = d;
					}
				}
			}
		}

		if (minD != numVertices + 1) {
			dist[v] = minD;
			ENQUEUE_DEVICE(v);
		}

		return minD;
	}

	__device__ __host__ ExcessType fluxoTotal() {
		return excess[numVertices - 1];
	}

	__host__ void bfs(Fila *bfsQ) {
		// double timeTotal = 0;
		// printf("global update!\n");
		double time1 = second();
		// Fila *bfsQ = new Fila;
		bfsQ->reset();

		for (int i = 0; i < numVertices; i++) {
			dist[i] = numVertices;
		}

		bfsQ->enfileirar(numVertices - 1);
		dist[numVertices - 1] = 0;

		// printf("bfs vai começar visitas!\n");
		while(!bfsQ->vazia()) {
			int v = bfsQ->desenfileirar();
			int novaDistancia = dist[v] + 1;
			Vertice *vertice = vertices + v;
			for (int i = 0; i < vertice->numAdjacentes; i++) {
				if (i < vertice->numAdjacentes) {
					Aresta *adj = arestas + vertice->adjacentes[i];
					if (resCap[adj->reversa] > 0) {
						if (dist[adj->to] == numVertices && adj->from == v) {
							dist[adj->to] = novaDistancia;
							bfsQ->enfileirar(adj->to);
						}
					}
				}
			}
		}

		dist[0] = numVertices;
		// printf("bfs timeTotal = %f\n", second() - time1);
	}

	__host__ void maxFlowInit() {
		double tempo1 = second();
		Fila filaBfs;
		bfs(&filaBfs);
		printf("bfs inicial tempo = %f\n", second() - tempo1);

		ativo[0] = ativo[numVertices - 1] = true;
		Fila *filaAtivos = new Fila;
		filaAtivos->init();

		for (int i = 0; i < vertices[0].numAdjacentes; i++) {
			// printf("from:%d, to:%d, resCap:%ld, excesso:%llu\n", ADJACENTE(0, i).from, ADJACENTE(0, i).to, ADJACENTE(0, i).resCap, excess[ADJACENTE(0, i).to]);
			Aresta *adjacente = &ADJACENTE(0, i);
			excess[0] += resCap[adjacente->id];
			Push(&ADJACENTE(0, i), filaAtivos);
		}

		// int contador = 0;
		// while(filaAtivos->tamanho > 0 && filaAtivos->tamanho < numVertices / 10) {
		// 	int v = filaAtivos->desenfileirar();
		// 	ativo[v] = false;
		// 	Discharge(v, filaAtivos);
		// 	if (contador++ >= nm * 0.4) {
		// 		bfs();
		// 		contador = 0;
		// 	}
		// }
		printf("tempo maxFlowInit = %f\n", second() - tempo1);
		printf("filaAtivos = %d\n", filaAtivos->tamanho);
	}

	_GrafoAloc alocaGrafoDevice() {
		cudaStream_t cs[100];
		for (int i = 0; i < 100; i++) {
			cudaStreamCreate(cs + i);
		}
		double tempo1 = second();
		Grafo *grafo_d, *grafo_tmp;
		Vertice *vertices_tmp;
		Fila *filaAtivos_tmp, *filaProx_tmp;
		cudaMallocHost(&grafo_tmp, sizeof(Grafo));
		cudaMallocHost(&vertices_tmp, sizeof(Vertice) * numVertices);
		cudaMallocHost(&filaAtivos_tmp, sizeof(Fila));
		cudaMallocHost(&filaProx_tmp, sizeof(Fila));
		memcpy(grafo_tmp, this, sizeof(Grafo));
		memcpy(vertices_tmp, vertices, sizeof(Vertice) * numVertices);
		cudaMalloc(&grafo_tmp->vertices, sizeof(Vertice) * numVertices);
		cudaMalloc(&grafo_tmp->excess, sizeof(ExcessType) * numVertices);
		cudaMalloc(&grafo_tmp->resCap, sizeof(CapType) * numArestas);
		cudaMalloc(&grafo_tmp->dist, sizeof(int) * numVertices);
		cudaMalloc(&grafo_tmp->arestas, sizeof(Aresta) * numArestas);
		cudaMalloc(&grafo_tmp->ativo, sizeof(int) * numVertices);
		cudaMalloc(&grafo_d, sizeof(Grafo));

		cudaMemcpyAsync(grafo_tmp->vertices, vertices_tmp, sizeof(Vertice) * numVertices, cudaMemcpyHostToDevice, cs[2]);
		cudaMemcpyAsync(grafo_tmp->excess, excess, sizeof(ExcessType) * numVertices, cudaMemcpyHostToDevice, cs[3]);
		cudaMemcpyAsync(grafo_tmp->resCap, resCap, sizeof(CapType) * numArestas, cudaMemcpyHostToDevice, cs[4]);
		cudaMemcpyAsync(grafo_tmp->dist, dist, sizeof(int) * numVertices, cudaMemcpyHostToDevice, cs[5]);
		cudaMemcpyAsync(grafo_tmp->arestas, arestas, sizeof(Aresta) * numArestas, cudaMemcpyHostToDevice, cs[6]);
		cudaMemcpyAsync(grafo_tmp->ativo, ativo, sizeof(int) * numVertices, cudaMemcpyHostToDevice, cs[7]);

		cudaMemcpyAsync(grafo_d, grafo_tmp, sizeof(Grafo), cudaMemcpyHostToDevice, cs[9]);
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess){
			// print the CUDA error message and exit
			printf("CUDA error na alocação: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
		printf("Tempo alocação = %f\n", second() - tempo1);
		GrafoAloc grafo_aloc;
		grafo_aloc.grafo_d = grafo_d;
		grafo_aloc.grafo_tmp = grafo_tmp;
		return grafo_aloc;
	}

	__host__ ExcessType fluxoTotalDevice(Grafo *grafo_h, Grafo *grafo_tmp) {
		bool *continuar;
		Fila filaBfs;
		ExcessType *fluxoTotal;
		double tempo1 = 0, tempo2 = 0, tempo3 = 0;
		unsigned long long i = 0;
		int stop = grafo_h->numVertices * 2;
		dim3 threads_per_block = 256;
		dim3 blocks = 256;
		cudaStream_t stream1, stream2;

		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		cudaHostAlloc((void **)&fluxoTotal, sizeof(ExcessType), cudaHostAllocMapped);
		cudaHostAlloc((void **)&continuar, sizeof(bool), cudaHostAllocMapped);
		*fluxoTotal = 0;

		printf("stop = %d\n", stop);
		cudaDeviceSynchronize();

		do {
			// printf("i:%d\n", i);
			*continuar = false;
			tempo1 = second();
			pushrelabel_kernel<<<blocks, threads_per_block>>>(this, fluxoTotal);
			check_fim<<<blocks, threads_per_block>>>(this, continuar);
			cudaDeviceSynchronize();
			tempo2 += second() - tempo1;
			if (i % (50) == 0) {
				tempo1 = second();
				cudaMemcpy(grafo_h->resCap, grafo_tmp->resCap, sizeof(CapType) * grafo_h->numArestas, cudaMemcpyDeviceToHost);
				grafo_h->bfs(&filaBfs);
				cudaMemcpy(grafo_tmp->dist, grafo_h->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyHostToDevice);
				tempo3 += second() - tempo1;
			}
			i++;
		} while (*continuar);
		check_flow<<<1, 1>>>(this, fluxoTotal);
		cudaThreadSynchronize();
		printf("tempo pushrelabel_kernel: %f  i:%llu\n", tempo2, i);
		printf("tempo bfs: %f\n", tempo3);
		// check for error
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess){
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			// exit(-1);
		}
		return *fluxoTotal;
	}


	void finalizar() {
		cudaFree(this);
		cudaDeviceReset();
	}

} Grafo;

__global__ void pushrelabel_kernel(Grafo *grafo, ExcessType *fluxoMaximo) {
	int rank = getRank();
	int tamanho = grafo->numVertices;

	while (rank < tamanho - 1 ) {
		if (grafo->ativo[rank]) {
			grafo->ativo[rank] = false;
			grafo->DischargeDevice(rank);
		}
		rank += blockDim.x * gridDim.x;
	}
}

__global__ void check_fim(Grafo *grafo, bool *continuar) {
	int rank = getRank();
	while (rank < grafo->numVertices -1 ) {
		if (grafo->ativo[rank]) {
			// printf("%d ativo [excess:%llu]\n", rank, grafo->excess[rank]);
			*continuar = true;
		}
		rank += blockDim.x * gridDim.x;
	}
}

__global__ void check_flow(Grafo *grafo, ExcessType *fluxoMaximo) {
	printf("Fluxo máximo gpu %d\n", grafo->fluxoTotal());
	*fluxoMaximo = grafo->fluxoTotal();
}