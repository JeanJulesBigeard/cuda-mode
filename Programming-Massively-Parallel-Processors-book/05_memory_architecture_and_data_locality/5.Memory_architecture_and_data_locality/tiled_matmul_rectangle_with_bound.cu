#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

// Kernel for matrix multiplication using shared memory tiling, with boundary
// checks
__global__ void matrixMulKernel(const float *M, const float *N, float *P, int j,
                                int k, int l) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // Block and thread indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the P element to work on
  int Row = by * TILE_WIDTH + ty; // in [0, j)
  int Col = bx * TILE_WIDTH + tx; // in [0, l)

  float Pvalue = 0.0f;

  // Number of phases needed is determined by the k dimension
  int phases = (k + TILE_WIDTH - 1) / TILE_WIDTH;

  // Loop over the M and N tiles required to compute the P element
  for (int ph = 0; ph < phases; ++ph) {

    // Load M tile: (Row < j) and (ph*TILE_WIDTH + tx < k)
    if ((Row < j) && ((ph * TILE_WIDTH + tx) < k))
      Mds[ty][tx] = M[Row * k + (ph * TILE_WIDTH + tx)];
    else
      Mds[ty][tx] = 0.0f;

    // Load N tile: (ph*TILE_WIDTH + ty < k) and (Col < l)
    if (((ph * TILE_WIDTH + ty) < k) && (Col < l))
      Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * l + Col];
    else
      Nds[ty][tx] = 0.0f;

    __syncthreads();

    // Compute partial products
    for (int i = 0; i < TILE_WIDTH; ++i) {
      Pvalue += Mds[ty][i] * Nds[i][tx];
    }

    __syncthreads();
  }

  // Write the computed value into the result matrix if within boundaries
  if (Row < j && Col < l)
    P[Row * l + Col] = Pvalue;
}

int main() {
  // Choose arbitrary dimensions that are not multiples of TILE_WIDTH:
  // TILE_WIDTH = 16, so let's pick dimensions not divisible by 16.
  int j = 203; // rows of M and P
  int k = 307; // cols of M and rows of N
  int l = 109; // cols of N and P

  // Allocate host memory
  float *h_M = (float *)malloc(j * k * sizeof(float));
  float *h_N = (float *)malloc(k * l * sizeof(float));
  float *h_P = (float *)malloc(j * l * sizeof(float));

  // Initialize host matrices with random values
  srand(0);
  for (int i = 0; i < j * k; i++) {
    h_M[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < k * l; i++) {
    h_N[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_M, *d_N, *d_P;
  cudaMalloc((void **)&d_M, j * k * sizeof(float));
  cudaMalloc((void **)&d_N, k * l * sizeof(float));
  cudaMalloc((void **)&d_P, j * l * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_M, h_M, j * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, k * l * sizeof(float), cudaMemcpyHostToDevice);

  // Compute grid dimensions
  int gridDimX = (l + TILE_WIDTH - 1) / TILE_WIDTH;
  int gridDimY = (j + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(gridDimX, gridDimY);

  // Launch kernel
  matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, j, k, l);

  // Copy result back to host
  cudaMemcpy(h_P, d_P, j * l * sizeof(float), cudaMemcpyDeviceToHost);

  // Print a small sub-matrix to verify output
  std::cout << "Sample of output matrix P (top-left 4x4):" << std::endl;
  for (int row = 0; row < 4 && row < j; row++) {
    for (int col = 0; col < 4 && col < l; col++) {
      std::cout << h_P[row * l + col] << "\t";
    }
    std::cout << std::endl;
  }

  // Clean up
  free(h_M);
  free(h_N);
  free(h_P);
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  return 0;
}
