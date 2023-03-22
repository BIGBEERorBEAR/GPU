#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <algorithm>

namespace image {

float* load(const std::string& filename, int* N, int* M, int* C, int C0)
{
    const unsigned char* bytes = stbi_load(filename.c_str(), N, M, C, C0);
    const auto size = (*N) * (*M) * (*C);
    float* data = (float*)malloc(size * sizeof(float));

    for(auto i = 0; i < size; ++i) 
        data[i] = float(bytes[i])/255;

    return data;
}

void save(const std::string& filename, int N, int M, int C, const float* data, int quality)
{
    const auto size = N * M * C;
    unsigned char* bytes = (unsigned char*)malloc(size * sizeof(unsigned char));

    for(auto i = 0; i < size; ++i) 
        bytes[i] = (unsigned char)(data[i]*255);

    stbi_write_jpg(filename.c_str(), N, M, C, bytes, quality);

    free(bytes);
}


} // namespace image
