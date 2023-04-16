#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include "timer.hpp"

// This example shows how to compute a bounding box
// for a set of points in two dimensions.

struct point2d {
    float x, y;

    __host__ __device__ point2d() : x(0), y(0) {}

    __host__ __device__ point2d(float _x, float _y) : x(_x), y(_y) {}
};

// bounding box type
struct bbox {
    // construct an empty box
    __host__ __device__ bbox() {}

    // construct a box from a single point
    __host__ __device__ bbox(const point2d &point)
        : lower_left(point), upper_right(point) {}

    // construct a box from a single point
    __host__ __device__ bbox &operator=(const point2d &point) {
        lower_left = point;
        upper_right = point;
        return *this;
    }

    // construct a box from a pair of points
    __host__ __device__ bbox(const point2d &ll, const point2d &ur)
        : lower_left(ll), upper_right(ur) {}

    point2d lower_left, upper_right;
};

// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct bbox_reduction : public thrust::binary_function<bbox, bbox, bbox> {
    __host__ __device__ bbox operator()(bbox a, bbox b) {
        // lower left corner
        point2d ll(thrust::min(a.lower_left.x, b.lower_left.x),
                   thrust::min(a.lower_left.y, b.lower_left.y));

        // upper right corner
        point2d ur(thrust::max(a.upper_right.x, b.upper_right.x),
                   thrust::max(a.upper_right.y, b.upper_right.y));

        return bbox(ll, ur);
    }
};

/* results
const size_t N = 1'0000'0000;
    GPU
        t_main - Execution time: 2.462 milli seconds
        bounding box (0.000000,0.000000) (1.000000,1.000000)

    CPU
        t_main - Execution time: 222.203 milli seconds
        bounding box (0.000000,0.000000) (1.000000,1.000000)

const size_t N = 1000'0000;
    GPU
        t_main - Execution time: 0.432 milli seconds
        bounding box (0.000000,0.000000) (1.000000,1.000000)

    CPU
        t_main - Execution time: 22.209 milli seconds
        bounding box (0.000000,0.000000) (1.000000,1.000000)
*/
int main(void) {
    const size_t N = 1'0000'0000;

    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // allocate storage for points
    std::vector<point2d> host_points(N);
    thrust::device_vector<point2d> points(N);

    // generate some random points in the unit square on the host
    for (size_t i = 0; i < N; i++) {
        float x = u01(rng);
        float y = u01(rng);
        host_points[i] = point2d(x, y);
    }

    // copy points from host to device
    thrust::copy(host_points.begin(), host_points.end(), points.begin());

    // initial bounding box contains first point
    bbox init = bbox(points[0], points[0]);

    // binary reduction operation
    bbox_reduction binary_op;

    // compute the bounding box for the point set
    bbox result{};
    double t_main = tictoc(
        [&]() {
            return thrust::reduce(points.begin(), points.end(), init,
                                  binary_op);
        },
        result);

    print_time(t_main, "t_main");

    // print output
    std::cout << "bounding box " << std::fixed;
    std::cout << "(" << result.lower_left.x << "," << result.lower_left.y
              << ") ";
    std::cout << "(" << result.upper_right.x << "," << result.upper_right.y
              << ")" << std::endl;

    return 0;
}