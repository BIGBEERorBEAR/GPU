#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

__global__ void add(float *x, float *y, int N) {
    int thr_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i = thr_id;

    while (i < N)
    {
        y[i] += x[i];
        i += stride;
    }
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e6;
    std::cout << "N = " << N << std::endl;

    float *x, *y;
    float *array_x, *array_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) 
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMalloc(&array_x, N * sizeof(float));
    cudaMalloc(&array_y, N * sizeof(float));
    cudaMemcpy(array_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(array_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    add<<<512,128>>>(array_x, array_y, N);
    cudaMemcpy(y, array_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());

    free(x);
    free(y);
    cudaFree(array_x);
    cudaFree(array_y);

    return 0;
}
