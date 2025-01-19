/********************************************************************
 * Compile with:
 *    nvcc -o matrixMul matrixMul.cu
 *
 * Run:
 *    ./matrixMul
 *
 * This code multiplies two square matrices A and B (both of dimension
 * width x width) on the GPU using a tiled approach with a coarse factor
 * in the column dimension. The result is stored in C on the device
 * and later copied back to the host for verification/printing.
 ********************************************************************/

#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time(NULL)
#include <cuda_runtime.h>
#include <iostream>

// ------------------------------------------------------------------
// Define constants for tiling dimensions and coarse factor unrolling
// ------------------------------------------------------------------
#define TILE_WIDTH 32
#define COARSE_FACTOR 4

/*************************************************************
 * Kernel: matrixMulKernel
 *
 * This kernel computes the product of two input matrices M and N
 * and writes the result to P using a tiled approach. Additionally,
 * it unrolls the computation in the "column" direction by a factor
 * of COARSE_FACTOR to improve performance.
 *
 * Parameters:
 *   M, N  - Pointers to the input matrices on the device
 *   P     - Pointer to the output matrix on the device
 *   width - Width (and height) of the square matrices M, N, and P
 *************************************************************/
__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
  // ----------------------------------------------------------
  // Create shared memory (tile) to hold sub-blocks of M and N
  // ----------------------------------------------------------
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // ----------------------------------------------------------
  // Each block is responsible for a TILE_WIDTH x TILE_WIDTH
  // region of the output matrix. bx, by identify which block
  // we are in. tx, ty identify which thread inside the block.
  // ----------------------------------------------------------
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // ----------------------------------------------------------
  // Calculate the row in the output matrix that this thread
  // will produce partial sums for, and the first column within
  // that row. Because we are unrolling in the column dimension
  // by COARSE_FACTOR, each thread will ultimately calculate
  // COARSE_FACTOR partial products in the same row.
  // ----------------------------------------------------------
  int row = by * TILE_WIDTH + ty;
  int colstart = bx * TILE_WIDTH + tx; // first column this thread handles

  // ----------------------------------------------------------
  // We will accumulate partial sums into this array, one for
  // each unrolled column. Initialize them to 0.
  // ----------------------------------------------------------
  float Pvalue[COARSE_FACTOR];
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    Pvalue[c] = 0.0f;
  }

  // ----------------------------------------------------------
  // Loop over all sub-tiles that cover the width of the matrix
  // Each sub-tile has size TILE_WIDTH in dimension. "ph" is
  // the tile index along the dimension of the matrix.
  // ----------------------------------------------------------
  for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
    // ------------------------------------------------------
    // Load the tile of M that corresponds to the row we are
    // responsible for, and the sub-tile 'ph'. All threads
    // in the block cooperatively load data into Mds.
    // ------------------------------------------------------
    if (row < width) {
      Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
    }
    __syncthreads(); // Make sure Mds is fully loaded before proceeding

    // ------------------------------------------------------
    // Unroll the multiplication over the column dimension
    // by COARSE_FACTOR. Each iteration c computes one chunk
    // of the output for a different column.
    // ------------------------------------------------------
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      // ----------------------------------------------
      // The column in the output for the c-th unrolled
      // iteration. We jump by TILE_WIDTH each time.
      // ----------------------------------------------
      int col = colstart + c * TILE_WIDTH;

      // ----------------------------------------------
      // Load the corresponding tile of N. Notice that
      // we offset by 'ph * TILE_WIDTH' in the row and
      // by 'col' in the column to get the correct sub-tile
      // of N. All threads in block cooperatively load.
      // ----------------------------------------------
      if (col < width) {
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
      }
      __syncthreads(); // Ensure Nds is fully loaded

      // ----------------------------------------------
      // Perform the actual multiply-add for the tile.
      // We iterate across the k dimension from 0 to
      // TILE_WIDTH - 1. We use data in Mds and Nds
      // that were cooperatively loaded.
      // ----------------------------------------------
      if (row < width && col < width) {
        for (int k = 0; k < TILE_WIDTH; ++k) {
          Pvalue[c] += Mds[ty][k] * Nds[k][tx];
        }
      }
      __syncthreads(); // Make sure we finish all math before next iteration
    }
  }

  // ----------------------------------------------------------
  // After accumulating all partial sums, write them back to
  // the output matrix P for each unrolled column index.
  // ----------------------------------------------------------
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int col = colstart + c * TILE_WIDTH;
    if (row < width && col < width) {
      P[row * width + col] = Pvalue[c];
    }
  }
}

// --------------------------------------------------
// Helper function to check for CUDA errors quickly
// --------------------------------------------------
void checkCudaError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// --------------------------------------------------
// Main program
// --------------------------------------------------
int main() {
  // --------------------------------------------
  // 1. Set matrix dimensions
  //    Ensure width is a multiple of TILE_WIDTH
  // --------------------------------------------
  int width = 128; // Example size; must be multiple of 32 (TILE_WIDTH)

  // --------------------------------------------
  // 2. Allocate host memory for matrices
  //    We have M, N as inputs and P as output
  // --------------------------------------------
  size_t size = width * width * sizeof(float);
  float *h_M = (float *)malloc(size);
  float *h_N = (float *)malloc(size);
  float *h_P = (float *)malloc(size);

  // --------------------------------------------
  // 3. Initialize host matrices with random data
  // --------------------------------------------
  srand((unsigned int)time(NULL));
  for (int i = 0; i < width * width; i++) {
    h_M[i] = static_cast<float>(rand()) / RAND_MAX; // Random float in [0,1)
    h_N[i] = static_cast<float>(rand()) / RAND_MAX; // Random float in [0,1)
  }

  // --------------------------------------------
  // 4. Allocate device memory
  // --------------------------------------------
  float *d_M, *d_N, *d_P;
  cudaMalloc((void **)&d_M, size);
  checkCudaError("cudaMalloc d_M failed");
  cudaMalloc((void **)&d_N, size);
  checkCudaError("cudaMalloc d_N failed");
  cudaMalloc((void **)&d_P, size);
  checkCudaError("cudaMalloc d_P failed");

  // --------------------------------------------
  // 5. Copy host data to device
  // --------------------------------------------
  cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
  checkCudaError("cudaMemcpy to d_M failed");
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
  checkCudaError("cudaMemcpy to d_N failed");

  // --------------------------------------------
  // 6. Set the grid and block dimensions
  //    - Each block is TILE_WIDTH x TILE_WIDTH
  //    - The grid is (width / TILE_WIDTH) x (width / TILE_WIDTH)
  // --------------------------------------------
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH, 1);

  // --------------------------------------------
  // 7. Launch the kernel
  // --------------------------------------------
  matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
  checkCudaError("Kernel launch failed");

  // --------------------------------------------
  // 8. Copy the result matrix back to host
  // --------------------------------------------
  cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
  checkCudaError("cudaMemcpy from d_P failed");

  // --------------------------------------------
  // (Optional) 9. Print part of the output
  //             to see if it's populated
  // --------------------------------------------
  std::cout << "Sample of resulting matrix P (top-left 8x8 block):\n";
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      std::cout << h_P[r * width + c] << " ";
    }
    std::cout << "\n";
  }

  // --------------------------------------------
  // 10. Free device and host memory
  // --------------------------------------------
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
  free(h_M);
  free(h_N);
  free(h_P);

  return 0;
}
