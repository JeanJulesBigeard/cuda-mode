#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define R 2
#define FILTER_SIZE (2 * R + 1)
#define FILTER_ELEMENTS (FILTER_SIZE * FILTER_SIZE)

// Declare constant memory for the filter (flattened 1D array).
__constant__ float d_F_const[FILTER_ELEMENTS];

__global__ void convolution_2D_constant_mem_kernel(const float *N, float *P,
                                                   int width, int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Guard: Ensure we are within image bounds.
  if (outCol >= width || outRow >= height)
    return;

  float Pvalue = 0.0f;
  // Loop over filter rows and columns.
  for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
    for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
      // Calculate the corresponding input image coordinate.
      int inRow = outRow - R + fRow;
      int inCol = outCol - R + fCol;
      // Only accumulate if the input coordinates are within bounds.
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue +=
            d_F_const[fRow * FILTER_SIZE + fCol] * N[inRow * width + inCol];
      }
    }
  }
  // Store the computed value.
  P[outRow * width + outCol] = Pvalue;
}

int main() {
  // Define image parameters.
  const int width = 16;
  const int height = 16;
  const int imageSize = width * height;
  const int filterElements = FILTER_ELEMENTS; // Equals 25 when R = 2.

  // Allocate host memory.
  float *h_N = new float[imageSize];
  float *h_F = new float[filterElements];
  float *h_P = new float[imageSize];

  // Initialize the image and filter with constant values for simplicity.
  for (int i = 0; i < imageSize; i++) {
    h_N[i] = 1.0f;
  }
  for (int i = 0; i < filterElements; i++) {
    h_F[i] = 1.0f;
  }

  // Allocate device memory.
  float *d_N, *d_P;
  cudaMalloc(&d_N, imageSize * sizeof(float));
  cudaMalloc(&d_P, imageSize * sizeof(float));

  // Copy image data from host to device.
  cudaMemcpy(d_N, h_N, imageSize * sizeof(float), cudaMemcpyHostToDevice);

  // Copy the filter from host to device constant memory.
  cudaMemcpyToSymbol(d_F_const, h_F, filterElements * sizeof(float));

  // Define execution configuration.
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the convolution kernel.
  convolution_2D_constant_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_N, d_P, width, height);
  cudaDeviceSynchronize();

  // Copy the result from device to host.
  cudaMemcpy(h_P, d_P, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the output matrix.
  std::cout << "Output Matrix:" << std::endl;
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      std::cout << h_P[row * width + col] << " ";
    }
    std::cout << std::endl;
  }

  // Free host and device memory.
  delete[] h_N;
  delete[] h_F;
  delete[] h_P;
  cudaFree(d_N);
  cudaFree(d_P);

  return 0;
}
