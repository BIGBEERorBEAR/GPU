#include <string>
#include <iostream>
#include "image.h"

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}


template <typename T>
__device__ inline T* get_ptr(T *img, int i, int j, int C, size_t pitch) {
	return img + (i * pitch) + (j * C);
}


__global__ void process(int N, int M, int C, int pitch, float* img)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if (i,j) is within the image bounds
    if (i < N && j < M) {
        // Get the address of the pixel (i,j)
        float* pixel_ptr = get_ptr(img, i, j, C, pitch);
        float gray = (pixel_ptr[0] + pixel_ptr[1] + pixel_ptr[2]) / 3;
        
        // Modify the pixel value
        // Example: set the pixel to red
        pixel_ptr[0] = gray; // red channel
        pixel_ptr[1] = gray; // green channel
        pixel_ptr[2] = gray; // blue channel
    }
}


int main(int argc, char const *argv[])
{
    const std::string filename = argc >= 2 ? argv[1] : "image.jpg";
    std::cout << "filename = " << filename << std::endl;

    int M = 0;
    int N = 0;
    int C = 0;
    float* img = image::load(filename, &N, &M, &C);
    std::cout << "N (columns, width) = " << N << std::endl;
    std::cout << "M (rows, height) = " << M << std::endl;
    std::cout << "C (channels, depth) = " << C << std::endl;

    float* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, C * N * sizeof(float), M);
    cudaMemcpy2D(d_img, pitch, img, C * N * sizeof(float), C * N * sizeof(float), M, cudaMemcpyHostToDevice);

    // Define the size of the grid and the block
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Call the kernel function to process the image
    process<<<grid, block>>>(N, M, C, pitch, d_img);

    // Copy the modified image back to the host
    cudaMemcpy2D(img, C * N * sizeof(float), d_img, pitch / sizeof(float), C * N * sizeof(float), M, cudaMemcpyDeviceToHost);

    image::save("res.jpg", N, M, C, img);

    free(img);

    return 0;
}
