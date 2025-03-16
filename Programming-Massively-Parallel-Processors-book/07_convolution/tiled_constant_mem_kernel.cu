#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define R 2
#define FILTER_SIZE (2 * R + 1)
#define FILTER_ELEMENTS (FILTER_SIZE * FILTER_SIZE)

#define IN_TILE_DIM 16
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * R)

// Declare constant memory for the filter (flattened 1D array).
__constant__ float d_F_const[FILTER_ELEMENTS];

__global__ void tiled_convolution_2D_constant_mem_kernel(const float *N,
                                                         float *P, int width,
                                                         int height) {
  // Beginning of the tile
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - R;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - R;

  // Loading the tile and handling ghost cells
  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }

  // Synchronize to ensure the tile is fully loaded
  __syncthreads();

  // Calculate output element indices within the tile
  int tileCol = threadIdx.x - R;
  int tileRow = threadIdx.y - R;

  // Only threads corresponding to valid output pixels perform the convolution
  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 &&
        tileRow < OUT_TILE_DIM) {
      float Pvalue = 0.0f;
      // Loop over the filter window
      for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
        for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
          // Access filter from constant memory (flattened array, row-major
          // order)
          float filter_value = d_F_const[fRow * FILTER_SIZE + fCol];
          Pvalue += filter_value * N_s[tileRow + fRow][tileCol + fCol];
        }
      }
      P[row * width + col] = Pvalue;
    }
  }
}

int main() {
  // Image dimensions
  const int width = 64;
  const int height = 64;
  const int size = width * height;
  const size_t bytes = size * sizeof(float);

  // Allocate host memory for input (N) and output (P)
  float *h_N = new float[size];
  float *h_P = new float[size];

  // Initialize the input image with random values
  srand(static_cast<unsigned>(time(nullptr)));
  for (int i = 0; i < size; ++i) {
    h_N[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Define a simple averaging filter
  float h_F[FILTER_ELEMENTS];
  for (int i = 0; i < FILTER_ELEMENTS; ++i) {
    h_F[i] = 1.0f / FILTER_ELEMENTS;
  }

  // Copy the filter to constant memory on the device
  cudaMemcpyToSymbol(d_F_const, h_F, FILTER_ELEMENTS * sizeof(float));

  // Allocate device memory for the input and output images
  float *d_N, *d_P;
  cudaMalloc((void **)&d_N, bytes);
  cudaMalloc((void **)&d_P, bytes);

  // Copy the input image to device memory
  cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
  dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
               (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  // Launch the kernel
  tiled_convolution_2D_constant_mem_kernel<<<dimGrid, dimBlock>>>(
      d_N, d_P, width, height);

  // Copy the result back to host memory
  cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

  // Print a few output values for verification.
  std::cout << "Sample output values:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << h_P[i] << " ";
  }
  std::cout << std::endl;

  // Free device memory
  cudaFree(d_N);
  cudaFree(d_P);

  // Free host memory
  delete[] h_N;
  delete[] h_P;

  return 0;
}
