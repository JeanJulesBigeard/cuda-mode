#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CHANNELS 3

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__
void colortoGrayscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row * width + col;
        // One can think of the RGB image having CHANNELS
        // times more columns than the grayscale image
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];     // red value
        unsigned char g = Pin[rgbOffset + 1]; // green value
        unsigned char b = Pin[rgbOffset + 2]; // blue value
        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main() {
    // Load the image using OpenCV
    cv::Mat inputImage = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open input image." << std::endl;
        return -1;
    }

    // Get image dimensions
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Ensure the image has 3 channels
    if (inputImage.channels() != 3) {
        std::cerr << "Error: Input image must have 3 channels (RGB)." << std::endl;
        return -1;
    }

    // Allocate host memory for the output image
    cv::Mat outputImage(height, width, CV_8UC1); // Grayscale image

    size_t rgb_image_size = width * height * CHANNELS * sizeof(unsigned char);
    size_t gray_image_size = width * height * sizeof(unsigned char);

    unsigned char *h_Pin = inputImage.ptr<unsigned char>(0);
    unsigned char *h_Pout = outputImage.ptr<unsigned char>(0);

    // Allocate device memory
    unsigned char *d_Pin, *d_Pout;
    cudaMalloc((void **)&d_Pin, rgb_image_size);
    cudaMalloc((void **)&d_Pout, gray_image_size);

    // Copy input data to device
    cudaMemcpy(d_Pin, h_Pin, rgb_image_size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y,
                 1);

    // Launch the kernel
    colortoGrayscaleConversion<<<dimGrid, dimBlock>>>(d_Pout, d_Pin, width, height);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
        // Free device memory before exiting
        cudaFree(d_Pin);
        cudaFree(d_Pout);
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(h_Pout, d_Pout, gray_image_size, cudaMemcpyDeviceToHost);

    // Save the output image
    if (!cv::imwrite("output.jpg", outputImage)) {
        std::cerr << "Error: Could not save output image." << std::endl;
        // Free device memory before exiting
        cudaFree(d_Pin);
        cudaFree(d_Pout);
        return -1;
    }

    // Free device memory
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    std::cout << "Grayscale conversion completed successfully. Output saved as 'output.jpg'." << std::endl;

    return 0;
}
