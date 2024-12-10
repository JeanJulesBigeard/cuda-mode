#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define BLUR_SIZE 10
#define CHANNELS 3

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        // For each channel
        for (int c = 0; c < CHANNELS; ++c) {
            int pixVal = 0;
            int pixels = 0;

            // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
                for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    // Verify we have a valid image pixel
                    if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        int offset = (curRow * w + curCol) * CHANNELS + c;
                        pixVal += in[offset];
                        ++pixels; // Keep track of the number of pixels in the avg
                    }
                } 
            }
            // Write our new pixel value out
            int offset = (row * w + col) * CHANNELS + c;
            out[offset] = (unsigned char)(pixVal / pixels);
        }
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
    cv::Mat outputImage(height, width, CV_8UC3);

    size_t rgb_image_size = width * height * CHANNELS * sizeof(unsigned char);

    unsigned char *h_in = inputImage.ptr<unsigned char>(0);
    unsigned char *h_out = outputImage.ptr<unsigned char>(0);

    // Allocate device memory
    unsigned char *d_in, *d_out;
    cudaMalloc((void **)&d_in, rgb_image_size);
    cudaMalloc((void **)&d_out, rgb_image_size);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, rgb_image_size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y,
                 1);

    // Launch the kernel with corrected argument order
    blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
        // Free device memory before exiting
        cudaFree(d_in);
        cudaFree(d_out);
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(h_out, d_out, rgb_image_size, cudaMemcpyDeviceToHost);

    // Save the output image
    if (!cv::imwrite("output.jpg", outputImage)) {
        std::cerr << "Error: Could not save output image." << std::endl;
        // Free device memory before exiting
        cudaFree(d_in);
        cudaFree(d_out);
        return -1;
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Blur completed successfully. Output saved as 'output.jpg'." << std::endl;

    return 0;
}
