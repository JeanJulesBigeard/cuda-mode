#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// Basic square Matmul with no boundary checks

#define TILE_WIDTH 16

// Kernel for matrix multiplication using shared memory tiling
__global__ void matrixMulKernel(const float *M, const float *N, float *P,
                                int Width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // Block and thread indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the P element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0.0f;

  // Loop over the M and N tiles required to compute the P element
  for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
    // Load the tile from M and N into shared memory
    Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

    __syncthreads();

    // Compute partial products
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    __syncthreads();
  }

  // Write the computed value into the result matrix
  P[Row * Width + Col] = Pvalue;
}

int main() {
  // Matrix dimension (must be multiple of TILE_WIDTH to avoid boundary checks),
  // the number of blocks in each dimension (N / TILE_WIDTH) matches the number
  // of tiles needed to cover the matrix.
  int N = 256; // Adjust as desired, but must be divisible by TILE_WIDTH

  // Size in bytes
  size_t size = N * N * sizeof(float);

  // Allocate host memory
  float *h_M = (float *)malloc(size);
  float *h_N = (float *)malloc(size);
  float *h_P = (float *)malloc(size);

  // Initialize host matrices with random values
  srand(0);
  for (int i = 0; i < N * N; i++) {
    h_M[i] = static_cast<float>(rand()) / RAND_MAX;
    h_N[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_M, *d_N, *d_P;
  cudaMalloc((void **)&d_M, size);
  cudaMalloc((void **)&d_N, size);
  cudaMalloc((void **)&d_P, size);

  // Copy data from host to device
  cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

  // Set up execution configuration
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);

  // Launch kernel
  matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, N);

  // Copy result back to host
  cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

  // Optional: Check a few elements for correctness or print a small sub-matrix
  std::cout << "Sample of output matrix (top-left 4x4):" << std::endl;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      std::cout << h_P[i * N + j] << "\t";
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
