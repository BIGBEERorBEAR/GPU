#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr auto block_dim = 256;  // constexpr equivalent to blockDim.x in CUDA kernel
constexpr auto block_count = 256; // constexpr equivalent to gridDim.x in CUDA kernel


int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e6;
    std::cout << "N = " << N << std::endl;

    float *x, *y;

    float host_expected_result = 0;
    float device_result = 0;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        y[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        host_expected_result += x[i] * y[i];
    }

    // ...

    std::cout << "host_expected_result = " << host_expected_result << std::endl;
    std::cout << "device_result = " << device_result << std::endl;

    free(x);
    free(y);
    
    return 0;
}