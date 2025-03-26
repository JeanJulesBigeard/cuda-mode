#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

// Define the tile dimensions:
// OUT_TILE_DIM: Number of output elements computed per block (per dimension)
// IN_TILE_DIM: Size of the shared memory tile including a halo of one element
// on each side
#define OUT_TILE_DIM 8
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

// Define the stencil coefficients (for a 7-point stencil)
#define c0 0.5f
#define c1 0.0833f
#define c2 0.0833f
#define c3 0.0833f
#define c4 0.0833f
#define c5 0.0833f
#define c6 0.0833f

// CUDA kernel performing 3D stencil computation using shared memory
__global__ void stencil_kernel_with_shared(const float *in, float *out,
                                           unsigned int N) {
  // Compute global indices for the current thread.
  // The subtraction by 1 accounts for the halo region.
  int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  // Allocate shared memory for the input tile (including halo regions).
  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

  // Load data from global memory to shared memory if the global index is valid.
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  } else {
    // If the index is out of bounds, set the shared memory element to zero.
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }

  // Synchronize to ensure that all threads have loaded their data into shared
  // memory.
  __syncthreads();

  // Compute the stencil only for valid output points (avoiding boundary issues)
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    // Additionally, ensure the thread is not working on the halo area within
    // shared memory.
    if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
        threadIdx.x < IN_TILE_DIM - 1) {
      // Compute the stencil: center element multiplied by c0 plus its six
      // neighbors
      out[i * N * N + j * N + k] =
          c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
          c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
          c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
          c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
          c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
          c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
          c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
    }
  }
}

int main() {
  // Define the problem size: a cube with N x N x N elements.
  const unsigned int N = 32; // Adjust as needed
  const size_t numElements = N * N * N;
  const size_t memSize = numElements * sizeof(float);

  // Allocate host memory for input and output arrays.
  float *h_in = (float *)malloc(memSize);
  float *h_out = (float *)malloc(memSize);

  // Initialize the input array with random values.
  srand(time(nullptr));
  for (size_t i = 0; i < numElements; ++i) {
    h_in[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory.
  float *d_in = nullptr;
  float *d_out = nullptr;
  cudaMalloc((void **)&d_in, memSize);
  cudaMalloc((void **)&d_out, memSize);

  // Copy input data from host to device.
  cudaMemcpy(d_in, h_in, memSize, cudaMemcpyHostToDevice);

  // Define block and grid dimensions.
  // Each block computes an output tile of size OUT_TILE_DIM^3,
  // but the block dimension (shared memory tile) is IN_TILE_DIM^3.
  dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
  // Calculate grid dimensions such that the entire domain is covered.
  dim3 gridDim((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
               (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
               (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  // Launch the CUDA kernel.
  stencil_kernel_with_shared<<<gridDim, blockDim>>>(d_in, d_out, N);
  cudaDeviceSynchronize();

  // Copy the computed result from device memory back to host memory.
  cudaMemcpy(h_out, d_out, memSize, cudaMemcpyDeviceToHost);

  // Optionally, print some output values for verification.
  std::cout << "Some output values:" << std::endl;
  for (unsigned int z = 1; z < N - 1; z += N / 4) {
    for (unsigned int y = 1; y < N - 1; y += N / 4) {
      for (unsigned int x = 1; x < N - 1; x += N / 4) {
        size_t idx = z * N * N + y * N + x;
        std::cout << "out[" << z << "][" << y << "][" << x
                  << "] = " << h_out[idx] << std::endl;
      }
    }
  }

  // Free device memory.
  cudaFree(d_in);
  cudaFree(d_out);

  // Free host memory.
  free(h_in);
  free(h_out);

  return 0;
}
