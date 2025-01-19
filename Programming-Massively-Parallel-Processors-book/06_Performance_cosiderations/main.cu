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
  // Create shared memory (tile) to hold sub-blocks of M and N.
  // Each tile is TILE_WIDTH x TILE_WIDTH elements.
  // ----------------------------------------------------------
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // ----------------------------------------------------------
  // Identify the block and thread within that block.
  // bx, by: Block indices in X, Y directions
  // tx, ty: Thread indices within the block
  // ----------------------------------------------------------
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // ----------------------------------------------------------
  // Calculate the output row this thread will handle, and
  // the first output column (colstart) for the unrolled columns.
  // row = the absolute row in the output matrix
  // colstart = the first column in the output matrix this
  //            thread will handle before moving by TILE_WIDTH
  //            steps for unrolling.
  // ----------------------------------------------------------
  int row = by * TILE_WIDTH + ty;
  int colstart = bx * TILE_WIDTH + tx; // first column for this thread

  // ----------------------------------------------------------
  // We will accumulate partial sums for COARSE_FACTOR columns.
  // Initialize them to 0.
  // ----------------------------------------------------------
  float Pvalue[COARSE_FACTOR];
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    Pvalue[c] = 0.0f;
  }

  // ----------------------------------------------------------
  // The tiled multiplication is divided into phases (ph).
  // We need width / TILE_WIDTH phases to cover the entire row
  // dimension of M (and corresponding column dimension of N).
  // ----------------------------------------------------------
  for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
    // ------------------------------------------------------
    // Load a tile of M: Mds[ty][tx] <- M[row, ph*TILE_WIDTH + tx]
    //
    //  - Because 'row' is constant for all threads in the same
    //    horizontal slice of the block, and 'tx' ranges from 0
    //    to TILE_WIDTH-1, threads in the same warp/block read
    //    consecutive elements in M's row.
    //  - This pattern ensures "coalesced" memory access for M
    //    because consecutive threads read consecutive floats.
    // ------------------------------------------------------
    if (row < width) {
      Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
    }
    __syncthreads(); // Ensure all M elements are loaded before proceeding

    // ------------------------------------------------------
    // Within each phase, we unroll the multiplication over
    // COARSE_FACTOR columns. Each iteration 'c' handles one
    // chunk (TILE_WIDTH columns apart in the final matrix).
    // ------------------------------------------------------
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      // --------------------------------------------------
      // The actual column index for this part of unrolling
      // is colstart + c*TILE_WIDTH.
      // --------------------------------------------------
      int col = colstart + c * TILE_WIDTH;

      // --------------------------------------------------
      // Load a tile of N: Nds[ty][tx] <- N[ph*TILE_WIDTH+ty, col]
      //
      //  - Here, 'ph*TILE_WIDTH + ty' represents the row in N.
      //  - 'col' is the column in N we are focusing on for
      //    this unrolled iteration.
      //  - Because 'ty' ranges from 0 to TILE_WIDTH-1 (for
      //    threads across the block in the Y direction), the
      //    addresses accessed are consecutive for each warp
      //    if 'col' is fixed. This ensures coalesced accesses
      //    for N as well.
      // --------------------------------------------------
      if (col < width) {
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
      }
      __syncthreads(); // Ensure N is fully loaded in shared memory

      // --------------------------------------------------
      // Multiply-accumulate with the data in shared memory.
      // We iterate over the k dimension from 0..TILE_WIDTH-1
      // for the sub-tile. We add the product of Mds[row][k]
      // and Nds[k][col] to the partial sum Pvalue[c].
      // --------------------------------------------------
      if (row < width && col < width) {
        for (int k = 0; k < TILE_WIDTH; ++k) {
          Pvalue[c] += Mds[ty][k] * Nds[k][tx];
        }
      }
      __syncthreads(); // Wait for all threads to finish before next iteration
    }
  }

  // ----------------------------------------------------------
  // After we've processed all the tiles for M and N, store
  // the computed partial sums (Pvalue[c]) to the output matrix.
  // Each thread writes COARSE_FACTOR columns in the same row.
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
  //    M, N as inputs, P as output
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
