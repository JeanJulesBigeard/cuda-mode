#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

// 2D convolution kernel with 1D array indexing for filter and output image.
__global__ void convolution_2D_basic_kernel(const float *N, const float *F,
                                            float *P, int r, int width,
                                            int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Guard: Ensure we are within image bounds.
  if (outCol >= width || outRow >= height)
    return;

  float Pvalue = 0.0f;
  int filterSize = 2 * r + 1;

  // Loop over filter rows and columns.
  for (int fRow = 0; fRow < filterSize; fRow++) {
    for (int fCol = 0; fCol < filterSize; fCol++) {
      // Calculate the corresponding input image coordinate.
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;
      // Only accumulate if the input coordinates are within bounds.
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue += F[fRow * filterSize + fCol] * N[inRow * width + inCol];
      }
    }
  }
  // Store the computed value.
  P[outRow * width + outCol] = Pvalue;
}

int main() {
  // Define image and filter parameters.
  const int width = 16;
  const int height = 16;
  const int r = 2; // Filter radius; filter size becomes 5x5.
  const int filterSize = 2 * r + 1;
  const int imageSize = width * height;
  const int filterElements = filterSize * filterSize;

  // Allocate host memory.
  float *h_N = new float[imageSize];
  float *h_F = new float[filterElements];
  float *h_P = new float[imageSize];

  // Initialize the image and filter with example values.
  // Here we use a constant value of 1.0f for simplicity.
  for (int i = 0; i < imageSize; i++) {
    h_N[i] = 1.0f;
  }
  for (int i = 0; i < filterElements; i++) {
    h_F[i] = 1.0f;
  }

  // Allocate device memory.
  float *d_N, *d_F, *d_P;
  cudaMalloc(&d_N, imageSize * sizeof(float));
  cudaMalloc(&d_F, filterElements * sizeof(float));
  cudaMalloc(&d_P, imageSize * sizeof(float));

  // Copy data from host to device.
  cudaMemcpy(d_N, h_N, imageSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, filterElements * sizeof(float), cudaMemcpyHostToDevice);

  // Define execution configuration.
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the convolution kernel.
  convolution_2D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_N, d_F, d_P, r, width, height);
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
  cudaFree(d_F);
  cudaFree(d_P);

  return 0;
}
