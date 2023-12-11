/*
    try test yourself!
    https://godbolt.org/z/x4G73af9a
*/

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <iostream>
#include <cmath>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <vector>

// 구조체를 정의하여 3D 포인트를 구면 좌표로 변환
struct ToSpherical {
    const float width, height;

    ToSpherical(float w, float h) : width(w), height(h) {}

    __device__ float2 operator()(float3 point) {
        float radius = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
        float azimuth = atan2f(point.y, point.x);
        float elevation = acosf(point.z / radius);

        // 방위각과 고도를 이미지 좌표로 변환
        float u = (azimuth + M_PI) / (2 * M_PI) * width;
        float v = elevation / M_PI * height;
        return make_float2(u, v);
    }
};

// 무작위 포인트 생성을 위한 구조체
struct RandomPointGenerator {
    unsigned int seed;

    RandomPointGenerator(unsigned int s) : seed(s) {}

    __device__ float3 operator()(const unsigned int n) const {
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> dist(-10.0, 10.0);
        rng.discard(n);

        return make_float3(dist(rng), dist(rng), dist(rng));
    }
};

void printMemoryUsage(const char* stage) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("Memory usage at %s: used = %f, free = %f MB, total = %f MB\n",
            stage, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}


int main() {

    // GPU 속성 가져오기
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    std::cout << "Using GPU: " << properties.name << std::endl;

    // CUDA 버전 가져오기
    int runtimeVer;
    cudaRuntimeGetVersion(&runtimeVer);
    std::cout << "CUDA Runtime Version: " << runtimeVer / 1000 << "." << (runtimeVer % 100) / 10 << std::endl;

    int driverVer;
    cudaDriverGetVersion(&driverVer);
    std::cout << "CUDA Driver Version: " << driverVer / 1000 << "." << (driverVer % 100) / 10 << std::endl;

    // 
    const size_t N = 5000000;
    const int width = 640, height = 480;
    cudaEvent_t start, stop;
    float totalTime = 0, elapsedTime = 0;

    // 이벤트 생성
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printMemoryUsage("before allocation");

    // 무작위 포인트 클라우드 생성
    thrust::device_vector<float3> d_points(N);

    cudaEventRecord(start);
    {
        thrust::transform(thrust::counting_iterator<unsigned int>(0),
                          thrust::counting_iterator<unsigned int>(N),
                          d_points.begin(),
                          RandomPointGenerator(time(nullptr)));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Point generation time: " << elapsedTime << " ms\n";
    totalTime += elapsedTime;

    // 포인트 클라우드를 구면 좌표로 변환
    thrust::device_vector<float2> d_image_coords(N);

    cudaEventRecord(start);
    {
        thrust::transform(d_points.begin(), d_points.end(), d_image_coords.begin(), ToSpherical(width, height));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Spherical transformation time: " << elapsedTime << " ms\n";
    totalTime += elapsedTime;

    printMemoryUsage("after allocation");

    // 결과를 호스트로 복사
    std::vector<float3> h_points(N);
    std::vector<float2> h_image_coords(N);
    cudaEventRecord(start);
    {
        thrust::copy(d_points.begin(), d_points.end(), h_points.begin());
        thrust::copy(d_image_coords.begin(), d_image_coords.end(), h_image_coords.begin());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Copying to host time: " << elapsedTime << " ms\n";
    totalTime += elapsedTime;

    // 호스트에서 결과 출력 (이 부분은 시간 측정 대상에서 제외)
    for (size_t i = 0; i < 100; i++) {
        std::cout << "Point " << i << ": (" << h_points[i].x << ", " << h_points[i].y << ", " << h_points[i].z << ")"
                  << " -> Image Coords: (" << h_image_coords[i].x << ", " << h_image_coords[i].y << ")" << std::endl;
    }

    // 전체 수행 시간 출력
    std::cout << "Total execution time: " << totalTime << " ms\n";

    // 이벤트 파괴
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


/*
ASM generation compiler returned: 0
Execution build compiler returned: 0
Program returned: 0

Using GPU: Tesla T4

CUDA Runtime Version: 12.2
CUDA Driver Version: 12.2

Memory usage at before allocation: used = 105.000000, free = 14825.562500 MB, total = 14930.562500 MB

Point generation time: 1.80637 ms
Spherical transformation time: 1.7665 ms

Memory usage at after allocation: used = 205.000000, free = 14725.562500 MB, total = 14930.562500 MB

Copying to host time: 21.1244 ms

Point 0: (-3.90097, -3.52748, 4.91908) -> Image Coords: (74.8831, 125.106)
Point 1: (-3.52748, 4.91908, 8.78057) -> Image Coords: (543.368, 92.2174)
Point 2: (4.91908, 8.78057, 6.95749) -> Image Coords: (427.985, 147.585)
Point 3: (8.78057, 6.95749, 5.02476) -> Image Coords: (388.253, 175.58)
Point 4: (6.95749, 5.02476, -9.84601) -> Image Coords: (383.71, 370.462)
Point 5: (5.02476, -9.84601, 3.39904) -> Image Coords: (208.065, 194.421)
Point 6: (-9.84601, 3.39904, -4.84678) -> Image Coords: (606.141, 306.542)
Point 7: (3.39904, -4.84678, 1.30517) -> Image Coords: (222.297, 206.845)
Point 8: (-4.84678, 1.30517, 1.62951) -> Image Coords: (613.206, 192.039)
Point 9: (1.30517, 1.62951, -1.81147) -> Image Coords: (411.212, 349.192)


*/

/* 설명 포인트
- cpu gpu 카피 비용이 대부분이다. 포인트 수 증가해도 Point generation time 과 Spherical transformation time 은 작음
*/
