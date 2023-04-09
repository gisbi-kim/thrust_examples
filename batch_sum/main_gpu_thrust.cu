#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

#include <chrono>

#include "config.hpp"
#include "timer.hpp"

void headline() {
    std::cout << "#####\nUsing GPU\nThrust version: " << THRUST_MAJOR_VERSION
              << "." << THRUST_MINOR_VERSION << "\n######\n"
              << std::endl;
}

int main(void) {
    headline();

    // 1. generate random data on the host
    std::cout << "Generating " << num_samples << " random numbers ...\n"
              << std::endl;
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<Number> dist(0, 9999);
    thrust::host_vector<Number> h_vec(num_samples);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]() { return dist(rng); });

    // 2. transfer the data to the device
    std::cout << "Transfer the data from host to the device (gpu)\n"
              << std::endl;
    thrust::device_vector<Number> d_vec = h_vec;

    // 3. compute sum on the device (gpu)
    auto do_sum = [&]() {
        // binary operation used to reduce values
        thrust::plus<Number> binary_op;
        Number init = 0;
        Number sum =
            thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
        return sum;
    };

    Number output = 0;
    double t_main = tictoc(do_sum, output);

    print_time(t_main, "t_main");
    std::cout << "sum is " << output << std::endl;

    return 0;
}
