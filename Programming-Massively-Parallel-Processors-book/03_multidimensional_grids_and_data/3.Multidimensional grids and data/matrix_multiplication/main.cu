#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 10 // Matrix size (N x N)

__global__ void matrixMultiply(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

void randomInitializeMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(float *matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int size = N * N * sizeof(float);

    // Host matrices
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices A and B with random values
    randomInitializeMatrix(h_A, N);
    randomInitializeMatrix(h_B, N);

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result (optional for small matrices)
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, N);
    std::cout << "Matrix C (Result):" << std::endl;
    printMatrix(h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}