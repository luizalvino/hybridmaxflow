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

using namespace std;

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
	if (v != numVertices - 1) { \
		if (!ativo[v]) { \
			ativo[v] = true; \
			Vertice *v_tmp = vertices + v; \
			Bucket *l_tmp = buckets + dist[v]; \
			aAdd(l_tmp, v_tmp); \
		} \
		if (dist[v] > aMax) aMax = dist[v]; \
		if (dist[v] < aMin) aMin = dist[v]; \
	} \
}
#define ENQUEUE_DEVICE(v) { \
	if (excess[v] > 0) { \
		ativo[v] = true; \
	} \
}

__device__ int getRank() {
	return blockIdx.y  * gridDim.x  * blockDim.z * blockDim.y * blockDim.x
        + blockIdx.x  * blockDim.z * blockDim.y * blockDim.x
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x
        + threadIdx.x;
}


#define ASIZE 16000

__device__ int dMaxD = 0;

typedef unsigned long long ExcessType;
typedef unsigned long CapType;

typedef struct _Vertice {
	int index;
	int inicial;
	int final;
	int current;
	struct _Vertice *bNext;
	struct _Vertice *bPrev;
} Vertice;

typedef struct bucketSt {
	Vertice *firstActive;
	Vertice *firstInactive;
} Bucket;

long i_dist;

#define aAdd(l,i)\
{\
  /* printf("aAdd = %d\n", i->index); */\
  i->bNext = l->firstActive;\
  l->firstActive = i;\
  i_dist = dist[i->index];\
  if (i_dist < aMin)\
    aMin = i_dist;\
  if (i_dist > aMax)\
    aMax = i_dist;\
  if (dMax < aMax)\
    dMax = aMax;\
}

/* i must be the first element */
#define aRemove(l,i)\
{\
  l->firstActive = i->bNext;\
}

Vertice *i_next, *i_prev;
#define iAdd(l,i)\
{\
  i_next = l->firstInactive;\
  i->bNext = i_next;\
  i->bPrev = sentinelNode;\
  i_next->bPrev = i;\
  l->firstInactive = i;\
}

#define iDelete(l,i)\
{\
  i_next = i->bNext;\
  if (l->firstInactive == i) {\
    l->firstInactive = i_next;\
    i_next->bPrev = sentinelNode;\
  }\
  else {\
    i_prev = i->bPrev;\
    i_prev->bNext = i_next;\
    i_next->bPrev = i_prev;\
  }\
}


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

__global__ void pushrelabel_kernel(int start, int stop, Vertice *vertices, int numVertices, Aresta *arestas, bool *ativo,
	ExcessType *excess, int *dist, CapType *resCap,	int idInicial, int idFinal);
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
	Bucket *buckets;
	Vertice *sentinelNode;
	long aMax;
	long aMin;
	long dMax;
	int *f1;
	int numVizinhosAnt;
	int numVizinhosProx;
	Mensagem *mensagensAnt;
	Mensagem *mensagensProx;
	std::vector<int> fronteiraAnt;
	std::vector<int> fronteiraProx;

	void init(int _numVertices, int _numArestas, int rank, int nprocs) {
		numVertices = _numVertices;
		numArestas = 0;
		vertices = new Vertice[_numVertices + 1];
		arestas = new Aresta[_numArestas * 2];
		excess = new ExcessType[numVertices];
		resCap = new CapType[_numArestas * 2];
		dist = new int[numVertices];
		ativo = new bool[numVertices];
		marcado = new bool[numVertices];
		buckets = new Bucket[numVertices+1];
		sentinelNode = vertices + numVertices;
		dMax = 0;
		f1 = (int *) calloc(numVertices, sizeof(int));

		int i;
		for (i = 0; i < numVertices; i++) {
			vertices[i].index = i;
			excess[i] = 0;
			dist[i] = 0;
			ativo[i] = false;
			marcado[i] = false;
		}
		sentinelNode->index = numVertices;

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
				novoGrafo->vertices[a.from].current = i;
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

	__host__ void Discharge(Vertice *v, long &k) {
		int stop = v->final;
		Bucket *l;
		long jD;
		do {
			int i;
			jD = dist[v->index] - 1;
			l = buckets + dist[v->index];
			for (i = v->current; i != stop; i++) {
				Aresta adj = arestas[i];
				if (resCap[adj.id] > 0) {
					if (jD == dist[adj.to]) {
						ExcessType delta = MIN(excess[adj.from], resCap[adj.id]);
						resCap[adj.id] -= delta;
						resCap[adj.reversa] += delta;

						if (adj.to != numVertices - 1) {
							Bucket *lj = buckets + jD;

							if (excess[adj.to] == 0) {
								Vertice *j = (vertices + adj.to);
								iDelete(lj, j);
								aAdd(lj, j);
								ativo[adj.to] = true;
							}
						}

						excess[adj.from] -= delta;
						excess[adj.to] += delta;
						if (!ativo[adj.to]) {
							ENQUEUE(adj.to);
						}
						if (excess[v->index] == 0) break;
					}
				}
			}
			if (excess[v->index] > 0) {
				Relabel(v, k);
				if (dist[v->index] == numVertices) break;
				if ( (l->firstActive == sentinelNode) && (l->firstInactive == sentinelNode)) {
					gap(l);
				}
				if (dist[v->index] == numVertices) break;
			} else {
				v->current = i;
				iAdd(l, v);
				break;
			}
		} while (1);
	}

	__host__ void Push(Aresta *a) {
		if (resCap[a->id] > 0) {
			ExcessType delta = MIN(excess[a->from], resCap[a->id]);
			#pragma omp critical
			{
				if (delta > 0 && dist[a->from] > dist[a->to]) {
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
	}

	__host__ void Relabel(Vertice *v, long &k) {
		int minD;
		dist[v->index] = minD = numVertices;

		k += 12;

		int i;
		int minA;
		for (i = v->inicial; i < v->final; i++) {
			k++;
			Aresta *adj = arestas + i;
			if (resCap[adj->id] > 0) {
				int d = dist[adj->to];
				if (d < minD) {
					minD = d;
					minA = i;
				}
			}
		}

		minD++;
		if (minD < numVertices) {
			dist[v->index] = minD;
			v->current = minA;
			ENQUEUE(v->index);
		}
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


	int gap(Bucket *emptyB)	{

	    Bucket *l;
	    Vertice *i;
	    long r; /* index of the bucket before l  */
	    int cc; /* cc = 1 if no nodes with positive excess before
			      the gap */

	    r = (emptyB - buckets) - 1;

	    /* set labels of nodes beyond the gap to "infinity" */
	    #pragma omp parallel
	    {
	    	#pragma omp for nowait
		    for (int k = 1; k <= dMax; k++) {
		    	l = emptyB + k;
		    //for (l = emptyB + 1; l <= buckets + dMax; l++) {
		        for (i = l -> firstInactive; i != sentinelNode; i = i -> bNext) {
		            dist[i->index] = numVertices;
		        }

		        l -> firstInactive = sentinelNode;
		    }
		}

	    cc = (aMin > r) ? 1 : 0;

	    dMax = r;
	    aMax = r;

	    return ( cc);

	}

	__host__ int bfs(bool cpu) {
		double time1 = second();
		int f1_size = 1;
		int aSize = 0;
		int numMarcados = 0;
		bool chegouFonte = false;
		int num_threads = 8;
		omp_set_dynamic(0);
		omp_set_num_threads(num_threads);
		int *bucket[num_threads];
		int bucket_size[num_threads];
		int bucket_first[num_threads];
		for (int i = 0; i < num_threads; ++i) {
			bucket[i] = (int *) malloc(sizeof(int) * numVertices);
			bucket_size[i] = 0;
		}
		// printf("num threads = %d\n", num_threads);

		std::fill(dist, dist + numVertices, numVertices);

		f1[0] = numVertices - 1;
		dist[numVertices - 1] = 0;

		if (cpu) {
			for (Bucket *l = buckets; l <= buckets + dMax; l++) {
				l -> firstActive = sentinelNode;
		        l -> firstInactive = sentinelNode;
		    }
		}

	    dMax = aMax = 0;
	    aMin = numVertices;

	    iAdd(buckets, (vertices + numVertices - 1));

		int k = 1;
		while(f1_size > 0) {
			//printf("f1 size %d\n", f1_size);
			int v, i, j, stop;
			Vertice *vertice;
			#pragma omp parallel for private(v,j,stop)
			for (i = 0; i < f1_size; i++) {
				int tid = omp_get_thread_num();
				for (v = f1[i], vertice = vertices + v, j = vertice->inicial, stop = vertice->final; j < stop; ++j) {
					Aresta adj = arestas[j];
					if (resCap[adj.reversa] > 0 && dist[adj.to] == numVertices && !marcado[adj.to]) {
						vertices[adj.to].current = vertices[adj.to].inicial;
						dist[adj.to] = k;
						int index = bucket_size[tid];
						bucket[tid][index] = adj.to;
						bucket_size[tid]++;
					}
				}
			}

			thrust::exclusive_scan(thrust::cpp::par, bucket_size, bucket_size + num_threads, bucket_first);

			#pragma omp parallel for private(j)
			for (i = 0; i < num_threads; i++) {
				for (j = 0; j < bucket_size[i]; j++) {
					f1[bucket_first[i] + j] = bucket[i][j];
				}
			}
			f1_size = bucket_first[num_threads - 1] + bucket_size[num_threads - 1];
			thrust::fill(thrust::cpp::par, bucket_size, bucket_size + num_threads, 0);
			k++;
		}

		if (*excessTotal > 0) {
			for (int i = 1; i < numVertices - 1; ++i) {
				if (dist[i] == numVertices && !marcado[i]) {
					numMarcados++;
					marcado[i] = true;
					ativo[i] = false;
					(*excessTotal) -= excess[i];
				}
				if (dist[i] != numVertices && cpu) {
					if (dist[i] > dMax) dMax = dist[i];
					if (ativo[i] && excess[i] > 0 && i != numVertices - 1) {
						aAdd((buckets + dist[i]), (vertices + i));
					} else {
						iAdd((buckets + dist[i]), (vertices + i));
					}
				}
			}
		}

		// dist[0] = numVertices;

		printf("tempo bfs = %f\n", second() - time1);
		// printf("numVisitados = %d | aSize = %d | marcados = %d | chegouFonte = %d | e[0] = %llu | e[n-1] = %llu | excessTotal = %llu\n", numVisitados, aSize, numMarcados, chegouFonte, excess[0], excess[numVertices - 1], *excessTotal);

		return (numMarcados == 0 && chegouFonte == 0) || aSize == 0;
	}

	__host__ void preparaBuckets() {
		#pragma omp parallel for
		for (int i = 0; i < numVertices; i++) {
			Bucket *l = buckets + i;
			l -> firstActive = sentinelNode;
	        l -> firstInactive = sentinelNode;
		}
		for (int i = 1; i < numVertices - 1; ++i) {
	    	if (dist[i] != numVertices) {
	    		if (dist[i] > dMax) {
	    			dMax = dist[i];
	    		}
	    		if (ativo[i] && excess[i] > 0 && i != numVertices - 1) {
	    			aAdd((buckets + dist[i]), (vertices + i));
	    		} else {
	    			iAdd((buckets + dist[i]), (vertices + i));
	    		}
	    	}
	    }
	}

	__host__ void maxFlowInit() {
		double tempo1 = second();
		// printf("bfs inicial tempo = %f\n", second() - tempo1);

		bfs(true);
		double tempo2 = second();
		std::sort(arestas, arestas + numArestas, ComparaArestaFromDist(dist));
		printf("tempo sort by dist = %f\n", second() - tempo2);

		for (Bucket *l = buckets; l <= buckets + numVertices - 1; l++) {
	        l -> firstActive = sentinelNode;
	        l -> firstInactive = sentinelNode;
	    }

		for (int i = vertices[0].inicial; i < vertices[0].final; ++i) {
			Aresta *adjacente = arestas + i;
			if (adjacente->to != 0 && resCap[adjacente->id] > 0) {
				CapType delta = resCap[adjacente->id];
				resCap[adjacente->id] -= delta;
				resCap[adjacente->reversa] += delta;
				excess[adjacente->to] += delta;
				(*excessTotal) += delta;
				ativo[adjacente->to] = true;
			}
		}

		Bucket *l = buckets + 1;

		aMax = 0;
		aMin = numVertices;

		for (int i = 0; i < numVertices; i++) {
			Vertice *v = vertices + i;
			if (i == numVertices - 1) {
				dist[i] = 0;
				iAdd(buckets, v);
				continue;
			}
			if (i == 0) {
				dist[i] = numVertices;
			} else {
				dist[i] = 1;
			}
			if (excess[i] > 0) {
				aAdd(l, v);
			} else {
				if (dist[i] < numVertices) {
					iAdd(l, v);
				}
			}
		}
		dMax = 1;
		// bfs();

		printf("tempo maxFlowInit = %f\n", second() - tempo1);
		printf("flow = %llu\n", excess[numVertices - 1]);
		//printf("filaAtivos = %d\n", f1_size);
	}


	__host__ int numAtivos() {
		int count = 0;
		for (int i = 0; i < numVertices; ++i) {
			if (excess[i] > 0 && ativo[i]) {
				count++;
			}
		}
		return count;
	}

	__host__ void pushrelabel_cpu() {
		Vertice *v;

		long k = 0;
		int numAtivos;
		while (aMax >= aMin) {
			Bucket *l = buckets + aMax;
			v = l->firstActive;

			if (v == sentinelNode) {
				aMax--;
			} else {
				aRemove(l, v);
				// printf("discharge = %d | excess %llu | firstActive %d\n", v->index, excess[v->index], l->firstActive->index);

			 	Discharge(v, k);
			 	ativo[v->index] = false;

			 	if (aMax < aMin) break;

			 	long nm = (long) 6 * numVertices + (numArestas / 2);
				if ( (k * 1.5) >= nm ) {
			 		k = 0;
			 		bfs(true);
			 		numAtivos = 0;
			 		for (int i = 0; i < numVertices; ++i) {
			 			if (excess[i] > 0) numAtivos++;
			 		}
			 		printf("numAtivos cpu = %d\n", numAtivos);
			 		if (numAtivos > ASIZE) break;
				}
			}
		}
	}

	_GrafoAloc alocaGrafoDevice() {
		cudaStream_t cs[10];
		for (int i = 0; i < 10; i++) {
			gpuErrchk( cudaStreamCreate(cs + i) );
		}
		double tempo1 = second();
		Grafo *grafo_tmp;
		Vertice *vertices_tmp;
		// Fila *filaAtivos_tmp, *filaProx_tmp;
		gpuErrchk( cudaMallocHost(&grafo_tmp, sizeof(Grafo)) );
		gpuErrchk( cudaMallocHost(&vertices_tmp, sizeof(Vertice) * numVertices) );
		// gpuErrchk( cudaMallocHost(&filaAtivos_tmp, sizeof(Fila)) );
		// gpuErrchk( cudaMallocHost(&filaProx_tmp, sizeof(Fila)) );
		memcpy(grafo_tmp, this, sizeof(Grafo));
		memcpy(vertices_tmp, vertices, sizeof(Vertice) * numVertices);
		gpuErrchk( cudaMalloc(&grafo_tmp->vertices, sizeof(Vertice) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->arestas, sizeof(Aresta) * numArestas) );
		gpuErrchk( cudaMalloc(&grafo_tmp->excess, sizeof(ExcessType) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->resCap, sizeof(CapType) * numArestas) );
		gpuErrchk( cudaMalloc(&grafo_tmp->dist, sizeof(int) * numVertices) );
		gpuErrchk( cudaMalloc(&grafo_tmp->ativo, sizeof(bool) * numVertices) );
		// gpuErrchk( cudaMalloc(&grafo_tmp->mensagensAnt, sizeof(Mensagem) * numVizinhosAnt) );
		// gpuErrchk( cudaMalloc(&grafo_tmp->mensagensProx, sizeof(Mensagem) * numVizinhosProx) );

		gpuErrchk( cudaMemcpyAsync(grafo_tmp->vertices, vertices_tmp, sizeof(Vertice) * numVertices, cudaMemcpyHostToDevice, cs[2]) );
		// gpuErrchk( cudaMemcpyAsync(grafo_tmp->excess, excess, sizeof(ExcessType) * numVertices, cudaMemcpyHostToDevice, cs[3]) );
		// gpuErrchk( cudaMemcpyAsync(grafo_tmp->resCap, resCap, sizeof(CapType) * numArestas, cudaMemcpyHostToDevice, cs[4]) );
		// gpuErrchk( cudaMemcpyAsync(grafo_tmp->dist, dist, sizeof(int) * numVertices, cudaMemcpyHostToDevice, cs[5]) );
		gpuErrchk( cudaMemcpyAsync(grafo_tmp->arestas, arestas, sizeof(Aresta) * numArestas, cudaMemcpyHostToDevice, cs[6]) );
		// gpuErrchk( cudaMemcpyAsync(grafo_tmp->ativo, ativo, sizeof(bool) * numVertices, cudaMemcpyHostToDevice, cs[7]) );
		// gpuErrchk( cudaMemcpyAsync(grafo_tmp->mensagensAnt, mensagensAnt, sizeof(Mensagem) * numVizinhosAnt, cudaMemcpyHostToDevice, cs[8]) );
		// gpuErrchk( cudaMemcpyAsync(grafo_tmp->mensagensProx, mensagensProx, sizeof(Mensagem) * numVizinhosProx, cudaMemcpyHostToDevice, cs[9]) );

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
		int num_blocos = ceil((double)(grafo_h->vertices_por_processo) / (256 * num_streams)) / 2;
		num_blocos = num_blocos > 1024 ? 1024 : num_blocos;
		printf("num_blocks = %d\n", num_blocos);
		dim3 threads_per_block = 256;
		dim3 blocks = num_blocos;
		int loop_size = 256 * num_blocos;
		// dim3 blocks = 128;

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		double tp1 = second();
		gpuErrchk( cudaHostAlloc((void **)&fluxoTotal, sizeof(ExcessType), cudaHostAllocMapped) );
		gpuErrchk( cudaHostAlloc((void **)&continuar, sizeof(bool), cudaHostAllocMapped) );
		*fluxoTotal = 0;

		printf("tempo inicial = %f\n", second() - tp1);
		cudaStream_t streams[4];
		for (int s = 0; s < 4; ++s) {
			gpuErrchk( cudaStreamCreate(&streams[s]) );
		}

		cudaDeviceSynchronize();

		tempoTotal = second();
		grafo_h->pushrelabel_cpu();
		int numAtivos = grafo_h->numAtivos();
		if (numAtivos > 0) {
			gpuErrchk( cudaMemcpyAsync(grafo_tmp->resCap, grafo_h->resCap, sizeof(CapType) * grafo_h->numArestas, cudaMemcpyHostToDevice, streams[0]) );
			gpuErrchk( cudaMemcpyAsync(grafo_tmp->ativo, grafo_h->ativo, sizeof(bool) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[1]) );
			gpuErrchk( cudaMemcpyAsync(grafo_tmp->excess, grafo_h->excess, sizeof(ExcessType) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[2]) );
			gpuErrchk( cudaMemcpyAsync(grafo_tmp->dist, grafo_h->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[3]) );
			gpuErrchk( cudaDeviceSynchronize() );
		}
		while(numAtivos > 0 && grafo_h->excess[grafo_h->numVertices - 1] < *grafo_h->excessTotal) {
			// printf("i:%d\n", i);
			*continuar = false;
			tempo1 = second();
			cudaMemset(&dMaxD, 0, sizeof(int));
			for (int l = 0; l < 98; ++l) {
				for (int start = grafo_h->idInicial; start <= grafo_h->idFinal; start += loop_size * 8) {
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[0]>>>(start, start + loop_size * (2), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal);
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[1]>>>(start + loop_size * 2, start + loop_size * (4), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal);
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[2]>>>(start + loop_size * 4, start + loop_size * (6), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal);
					pushrelabel_kernel<<<blocks, threads_per_block, 0, streams[3]>>>(start + loop_size * 6, start + loop_size * (8), grafo_tmp->vertices,
						grafo_h->numVertices, grafo_tmp->arestas, grafo_tmp->ativo, grafo_tmp->excess, grafo_tmp->dist, grafo_tmp->resCap,
						grafo_h->idInicial, grafo_h->idFinal);
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
				gpuErrchk( cudaMemcpyAsync(grafo_h->excess, grafo_tmp->excess, sizeof(ExcessType) * grafo_h->numVertices, cudaMemcpyDeviceToHost, streams[2]) );
				gpuErrchk( cudaDeviceSynchronize() );

				tempo1 = second();
				grafo_h->bfs(false);
				tempo3 += second() - tempo1;
				numAtivos = grafo_h->numAtivos();
				printf("numAtivos = %d\n", numAtivos);
				bool cpu = false;
				if (numAtivos < ASIZE) {
					cpu = true;
					grafo_h->preparaBuckets();
					grafo_h->pushrelabel_cpu();
					numAtivos = grafo_h->numAtivos();
					if (numAtivos == 0) break;
				}
				if (cpu) {
					gpuErrchk( cudaMemcpyAsync(grafo_tmp->resCap, grafo_h->resCap, sizeof(CapType) * grafo_h->numArestas, cudaMemcpyHostToDevice, streams[1]) );
					gpuErrchk( cudaMemcpyAsync(grafo_tmp->ativo, grafo_h->ativo, sizeof(bool) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[2]) );
					gpuErrchk( cudaMemcpyAsync(grafo_tmp->excess, grafo_h->excess, sizeof(ExcessType) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[3]) );
				}
				printf("rodando em gpu ou terminando\n");

				/*
					Se achou o fluxo máximo, encerra e notifica os outros processos
				*/
				bool sair;

				sair = grafo_h->achouFluxoMaximo() || (nproc > 1 && grafo_h->excess[grafo_h->numVertices-1] > 0 && global_enviou == 0);

				if (sair) {
					break;
				}

				tempoCopia = second();

				gpuErrchk( cudaMemcpyAsync(grafo_tmp->dist, grafo_h->dist, sizeof(int) * grafo_h->numVertices, cudaMemcpyHostToDevice, streams[0]) );
				gpuErrchk( cudaDeviceSynchronize() );
			}
			tempo4 += second() - tempoMsg;

			i++;
		}
		// grafo_h->bfs(false);
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
		printf("excess[%d] = %llu | excessTotal = %llu\n", numVertices - 1, excess[numVertices - 1], *excessTotal);
		return excess[0] + excess[numVertices - 1] >= *excessTotal;
	}

	void finalizar() {
		cudaDeviceReset();
	}

} Grafo;

#define VERTICES_POR_THREAD 2

__global__ void pushrelabel_kernel(int start, int stop, Vertice *vertices, int numVertices, Aresta *arestas, bool *ativo,
	ExcessType *excess, int *dist, CapType *resCap,	int idInicial, int idFinal) {
	const int rankBase = getRank() * VERTICES_POR_THREAD + start;

	#pragma unroll
	for (int t = 0; t < VERTICES_POR_THREAD; ++t) {
		int rank = rankBase + t;
		if (rank <= stop && rank <= idFinal) {
			if (ativo[rank] && excess[rank] > 0) {
				ativo[rank] = false;
				int v = rank;
				int jD = dist[v];
				int current = vertices[v].inicial;
				int stopAdj = vertices[v].final;
				while(excess[v] > 0 && jD < numVertices) {
					int minD = numVertices;
					#pragma unroll
					for (int i = current; i < stopAdj; ++i) {
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
									atomicAdd((unsigned long long *) &excess[a.to], delta);
									if (!ativo[a.to] && a.to != numVertices - 1) {
										ENQUEUE_DEVICE(a.to);
									}
								}
							}
							if (excess[v] == 0){break;}
						}
					}

					if (excess[v] > 0 && minD < numVertices) {
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
