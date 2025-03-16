#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define TILE_DIM 32
#define R 2
#define FILTER_SIZE (2 * R + 1)
#define FILTER_ELEMENTS (FILTER_SIZE * FILTER_SIZE)

// Declare constant memory for the filter as a 2D array.
__constant__ float F_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P,
                                                             int width,
                                                             int height) {
  // Compute global row and column indices for this thread.
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  // Allocate shared memory for the current tile.
  __shared__ float N_s[TILE_DIM][TILE_DIM];

  // Load data into shared memory if within image bounds.
  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  // Only threads corresponding to valid output pixels proceed.
  if (row < height && col < width) {
    float Pvalue = 0.0f;
    // Loop over the filter window.
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
      for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
        // Compute the corresponding shared memory indices.
        int shared_row = threadIdx.y - R + fRow;
        int shared_col = threadIdx.x - R + fCol;
        if (shared_row >= 0 && shared_row < TILE_DIM && shared_col >= 0 &&
            shared_col < TILE_DIM) {
          // If within the tile, use the shared memory.
          Pvalue += F_c[fRow][fCol] * N_s[shared_row][shared_col];
        } else {
          // Otherwise, fall back to global memory.
          int global_row = row - R + fRow;
          int global_col = col - R + fCol;
          if (global_row >= 0 && global_row < height && global_col >= 0 &&
              global_col < width) {
            Pvalue += F_c[fRow][fCol] * N[global_row * width + global_col];
          }
        }
      }
    }
    P[row * width + col] = Pvalue;
  }
}

int main() {
  // Image dimensions.
  const int width = 64;
  const int height = 64;
  const int size = width * height;
  size_t bytes = size * sizeof(float);

  // Allocate host memory for input and output images.
  float *h_N = new float[size];
  float *h_P = new float[size];

  // Initialize the input image with random values.
  srand(static_cast<unsigned>(time(nullptr)));
  for (int i = 0; i < size; ++i) {
    h_N[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Define a simple averaging filter.
  float h_F[FILTER_SIZE][FILTER_SIZE];
  for (int i = 0; i < FILTER_SIZE; ++i) {
    for (int j = 0; j < FILTER_SIZE; ++j) {
      h_F[i][j] = 1.0f / FILTER_ELEMENTS;
    }
  }

  // Copy the filter to constant memory F_c.
  cudaMemcpyToSymbol(F_c, h_F, FILTER_SIZE * FILTER_SIZE * sizeof(float));

  // Allocate device memory.
  float *d_N, *d_P;
  cudaMalloc((void **)&d_N, bytes);
  cudaMalloc((void **)&d_P, bytes);

  // Copy input image to device.
  cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

  // Set grid and block dimensions.
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
            (height + TILE_DIM - 1) / TILE_DIM);

  // Launch the convolution kernel.
  convolution_cached_tiled_2D_const_mem_kernel<<<grid, block>>>(d_N, d_P, width,
                                                                height);
  cudaDeviceSynchronize();

  // Copy the result back to host.
  cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

  // Print a few output values for verification.
  std::cout << "Sample output values:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << h_P[i] << " ";
  }
  std::cout << std::endl;

  // Free device and host memory.
  cudaFree(d_N);
  cudaFree(d_P);
  delete[] h_N;
  delete[] h_P;

  return 0;
}
