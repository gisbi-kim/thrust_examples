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
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_points.begin(),
                      RandomPointGenerator(time(nullptr)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Point generation time: " << elapsedTime << " ms\n";
    totalTime += elapsedTime;

    // 포인트 클라우드를 구면 좌표로 변환
    thrust::device_vector<float2> d_image_coords(N);

    cudaEventRecord(start);
    thrust::transform(d_points.begin(), d_points.end(), d_image_coords.begin(), ToSpherical(width, height));
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
    thrust::copy(d_points.begin(), d_points.end(), h_points.begin());
    thrust::copy(d_image_coords.begin(), d_image_coords.end(), h_image_coords.begin());
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

ASM generation compiler returned: 0
Execution build compiler returned: 0
Program returned: 0
Memory usage at before allocation: used = 105.000000, free = 14825.562500 MB, total = 14930.562500 MB
Point generation time: 1.88826 ms
Spherical transformation time: 1.83283 ms
Memory usage at after allocation: used = 205.000000, free = 14725.562500 MB, total = 14930.562500 MB
Copying to host time: 473.374 ms
Point 0: (1.54322, -7.14025, -6.97402) -> Image Coords: (181.681, 356.458)
Point 1: (-7.14025, -6.97402, -3.01801) -> Image Coords: (78.8004, 284.864)
Point 2: (-6.97402, -3.01801, -2.43688) -> Image Coords: (41.601, 287.414)
Point 3: (-3.01801, -2.43688, 9.51612) -> Image Coords: (69.1892, 59.1389)
Point 4: (-2.43688, 9.51612, -7.5907) -> Image Coords: (505.535, 340.519)
Point 5: (9.51612, -7.5907, 9.08719) -> Image Coords: (251.416, 142.021)
Point 6: (-7.5907, 9.08719, 7.63673) -> Image Coords: (550.885, 152.478)
Point 7: (9.08719, 7.63673, -7.61795) -> Image Coords: (391.188, 327.178)
Point 8: (7.63673, -7.61795, -5.92422) -> Image Coords: (240.125, 316.737)
Point 9: (-7.61795, -5.92422, -7.91345) -> Image Coords: (67.3262, 344.939)
Point 10: (-5.92422, -7.91345, 9.83294) -> Image Coords: (94.5431, 120.406)
Point 11: (-7.91345, 9.83294, 5.82871) -> Image Coords: (549.025, 173.9)
Point 12: (9.83294, 5.82871, -2.1359) -> Image Coords: (374.504, 268.224)
Point 13: (5.82871, -2.1359, -1.89917) -> Image Coords: (284.222, 285.362)
Point 14: (-2.1359, -1.89917, 5.371) -> Image Coords: (74.0309, 74.7179)
Point 15: (-1.89917, 5.371, 3.77571) -> Image Coords: (514.619, 150.573)
Point 16: (5.371, 3.77571, -2.52031) -> Image Coords: (382.412, 296.002)
Point 17: (3.77571, -2.52031, 2.00765) -> Image Coords: (260.047, 176.38)

*/

/* 설명 포인트
- cpu gpu 카피 비용이 대부분이다. 포인트 수 증가해도 Point generation time 과 Spherical transformation time 은 작음
*/
