#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

#include <chrono>
#include <iomanip>
#include <random>

#include "config.hpp"
#include "timer.hpp"

void headline()
{
    std::cout << "#####\nUsing GPU\nThrust version: " << THRUST_MAJOR_VERSION
              << "." << THRUST_MINOR_VERSION << "\n######\n"
              << std::endl;
}

class Point
{
public:
    double x;
    double y;
    double z;

    // Default constructor
    __host__ __device__ Point() : x(0.), y(0.), z(0.) {}

    // Parameterized constructor
    __host__ __device__ Point(double x, double y, double z)
        : x(x), y(y), z(z) {}

    // Addition operator
    __host__ __device__ Point operator+(const Point &other) const
    {
        return Point(this->x + other.x, this->y + other.y, this->z + other.z);
    }
};

struct PointAddition : public thrust::binary_function<Point, Point, Point>
{
    __host__ __device__ Point operator()(const Point &a, const Point &b) const
    {
        return a + b;
    }
};

class RandomPointGenerator
{
public:
    RandomPointGenerator(double mu, double std) : mu(mu), std(std) {}

    __host__ Point operator()() const
    {
        std::vector<double> xyz{};
        for (int i = 0; i < 3; ++i)
        {
            unsigned int seed = std::random_device{}();
            thrust::default_random_engine rng(seed);
            thrust::normal_distribution<double> dist(mu, std);
            xyz.push_back(dist(rng));
        }

        double x = xyz.at(0);
        double y = xyz.at(1);
        double z = xyz.at(2);

        if (verbose)
        {
            std::cout << " Generate a point - ";
            std::cout << "x: " << x << ", ";
            std::cout << "y: " << y << ", ";
            std::cout << "z: " << z << std::endl;
        }

        return Point(x, y, z);
    }

private:
    double mu;
    double std;
    const bool verbose = false;
};

int main(void)
{
    headline();

    // 1. generate random data on the host
    std::cout << "Generating " << num_samples << " random points ...\n"
              << std::endl;

    RandomPointGenerator rng(0.0, 1.5);
    thrust::host_vector<Point> h_vec(num_samples);
    thrust::generate(h_vec.begin(), h_vec.end(), rng);

    // 2. transfer the data to the device
    std::cout << "Transfer the data from host to the device (gpu)\n"
              << std::endl;
    thrust::device_vector<Point> d_vec = h_vec;

    // 3. compute sum on the device (gpu)
    auto do_sum = [&]()
    {
        PointAddition add_op;
        Point origin{};
        Point averaged_point =
            thrust::reduce(d_vec.begin(), d_vec.end(), origin, add_op);
        return averaged_point;
    };

    Point averaged_point{};
    double t_main = tictoc(do_sum, averaged_point);

    print_time(t_main, "t_main");
    std::cout << std::setprecision(9) << "The averaged_point is "
              << averaged_point.x << ", " << averaged_point.y << ", "
              << averaged_point.z << std::endl;

    return 0;
}
