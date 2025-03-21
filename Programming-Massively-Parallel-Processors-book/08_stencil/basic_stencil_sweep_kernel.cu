#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

// Define the stencil coefficients
#define c0 0.5f
#define c1 0.0833f
#define c2 0.0833f
#define c3 0.0833f
#define c4 0.0833f
#define c5 0.0833f
#define c6 0.0833f

// 3D 7-point stencil kernel
__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
  // Compute 3D indices; note the corrected multiplication for k.
  unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  // Apply stencil only if within bounds (avoid the outer "ghost" cells)
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] +
                                 c1 * in[i * N * N + j * N + (k - 1)] +
                                 c2 * in[i * N * N + j * N + (k + 1)] +
                                 c3 * in[i * N * N + (j - 1) * N + k] +
                                 c4 * in[i * N * N + (j + 1) * N + k] +
                                 c5 * in[(i - 1) * N * N + j * N + k] +
                                 c6 * in[(i + 1) * N * N + j * N + k];
  }
}

int main() {
  // Problem size: using an N x N x N grid
  unsigned int N = 32;
  size_t size = N * N * N * sizeof(float);

  // Allocate host memory
  float *h_in = new float[N * N * N];
  float *h_out = new float[N * N * N];

  // Initialize host input array with random values
  srand(static_cast<unsigned>(time(0)));
  for (unsigned int idx = 0; idx < N * N * N; idx++) {
    h_in[idx] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copy input data from host to device
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // Configure kernel launch parameters:
  // Use a 3D block of threads (8x8x8) and calculate the grid dimensions
  // accordingly.
  dim3 block(8, 8, 8);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y,
            (N + block.z - 1) / block.z);

  // Launch the kernel
  stencil_kernel<<<grid, block>>>(d_in, d_out, N);

  // Wait for the kernel to finish
  cudaDeviceSynchronize();

  // Copy result back from device to host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  // Output a sample result (printing the value at the center of the grid)
  unsigned int center = (N / 2) * N * N + (N / 2) * N + (N / 2);
  std::cout << "Result at center: " << h_out[center] << std::endl;

  // Free device and host memory
  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out;

  return 0;
}
