#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// Kernel for matrix multiplication using shared memory tiling, with boundary
// checks
__global__ void matrixMulKernel(const float *M, const float *N, float *P, int j,
                                int k, int l, int tile_width) {
  extern __shared__ float sMem[];
  // Mds and Nds point into the same dynamically allocated shared memory
  float *Mds = sMem;
  float *Nds = sMem + tile_width * tile_width;

  // Block and thread indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the P element to work on
  int Row = by * tile_width + ty; // in [0, j)
  int Col = bx * tile_width + tx; // in [0, l)

  float Pvalue = 0.0f;

  // Number of phases needed is determined by the k dimension
  int phases = (k + tile_width - 1) / tile_width;

  for (int ph = 0; ph < phases; ++ph) {
    // Load M tile
    if (Row < j && (ph * tile_width + tx) < k)
      Mds[ty * tile_width + tx] = M[Row * k + (ph * tile_width + tx)];
    else
      Mds[ty * tile_width + tx] = 0.0f;

    // Load N tile
    if ((ph * tile_width + ty) < k && Col < l)
      Nds[ty * tile_width + tx] = N[(ph * tile_width + ty) * l + Col];
    else
      Nds[ty * tile_width + tx] = 0.0f;

    __syncthreads();

    // Compute partial products
    for (int i = 0; i < tile_width; ++i) {
      Pvalue += Mds[ty * tile_width + i] * Nds[i * tile_width + tx];
    }

    __syncthreads();
  }

  // Write the computed value into the result matrix if within boundaries
  if (Row < j && Col < l)
    P[Row * l + Col] = Pvalue;
}

int main() {
  // Choose arbitrary dimensions not multiples of tile_width
  int j = 203; // rows of M and P
  int k = 307; // cols of M and rows of N
  int l = 109; // cols of N and P

  // Get device properties
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);

  // Dynamically select tile_width based on available shared memory
  // We'll try some candidates. In practice, you'd have a more sophisticated
  // method.
  int candidateTileWidths[] = {32, 16, 8, 4};
  int tile_width = 0;

  for (int tw : candidateTileWidths) {
    size_t needed = 2 * tw * tw * sizeof(float);
    if (needed <= devProp.sharedMemPerBlock) {
      tile_width = tw;
      break;
    }
  }

  if (tile_width == 0) {
    std::cerr << "No suitable tile_width found for this device's shared memory "
                 "constraints."
              << std::endl;
    return EXIT_FAILURE;
  }

  // Print the chosen tile width
  std::cout << "Chosen tile_width: " << tile_width << std::endl;

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
  int gridDimX = (l + tile_width - 1) / tile_width;
  int gridDimY = (j + tile_width - 1) / tile_width;
  dim3 dimBlock(tile_width, tile_width);
  dim3 dimGrid(gridDimX, gridDimY);

  // Calculate dynamic shared memory size based on chosen tile_width
  size_t size = 2 * tile_width * tile_width * sizeof(float);

  // Launch kernel with dynamic shared memory
  matrixMulKernel<<<dimGrid, dimBlock, size>>>(d_M, d_N, d_P, j, k, l,
                                               tile_width);

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
