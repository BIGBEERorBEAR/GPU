#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr auto block_dim = 256;  // constexpr equivalent to blockDim.x in CUDA kernel
constexpr auto block_count = 256; // constexpr equivalent to gridDim.x in CUDA kernel

__global__ void dot(int n, const float *x, const float *y, float* res) {
    __shared__ float buffer[block_dim];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0;

    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        local_sum += x[i] * y[i];
    }

    buffer[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buffer[tid] += buffer[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        res[blockIdx.x] = buffer[0];
    }
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e6;
    std::cout << "N = " << N << std::endl;

    float *x, *y, *d_x, *d_y, *d_res, *h_res;

    float host_expected_result = 0;
    float device_result = 0;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    h_res = (float*)malloc(block_count * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        y[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        host_expected_result += x[i] * y[i];
    }

    float *device_x, *device_y, *device_res;
    CUDA_CHECK(cudaMalloc(&device_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_res, block_count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(device_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<block_count, block_dim>>>(N, device_x, device_y, device_res);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *device_result_ptr = (float*)malloc(block_count * sizeof(float));
    CUDA_CHECK(cudaMemcpy(device_result_ptr, device_res, block_count * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < block_count; i++) {
        device_result += device_result_ptr[i];
    }

    CUDA_CHECK(cudaFree(device_x));
    CUDA_CHECK(cudaFree(device_y));
    CUDA_CHECK(cudaFree(device_res));

    free(device_result_ptr);

    std::cout << "host_expected_result = " << host_expected_result << std::endl;
    std::cout << "device_result = " << device_result << std::endl;

    free(x);
    free(y);
    
    return 0;
}