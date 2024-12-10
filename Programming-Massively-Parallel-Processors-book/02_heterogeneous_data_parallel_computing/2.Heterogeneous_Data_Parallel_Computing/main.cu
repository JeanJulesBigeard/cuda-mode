#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i]; 
    }
}

// Compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Kernel invocation
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

    // Copy result back to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    const int N = 3; // Size of the vectors
    float A[N] = {1, 2, 3};
    float B[N] = {3, 2, 1};
    float C[N]; // To store the result

    vecAdd(A, B, C, N);

    // Print the result
    std::cout << "Result of vector addition: ";
    for (int i = 0; i < N; i++){
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}