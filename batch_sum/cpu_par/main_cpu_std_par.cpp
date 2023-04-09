#include <algorithm>  // generate 함수를 사용하기 위한 헤더 파일
#include <execution>
#include <iostream>
#include <numeric>  // accumulate 함수를 사용하기 위한 헤더 파일
#include <random>
#include <vector>

#include "config.hpp"
#include "timer.hpp"

void headline() {
    std::cout << "#####\nUsing CPU and C++17's par execution policy"
              << "\n######\n"
              << std::endl;
}

int main(void) {
    headline();

    // 1. generate random data on the host
    std::cout << "Generating " << num_samples << " random numbers ...\n"
              << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9999);
    std::vector<Number> vec(num_samples);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });

    // 2. compute sum at the host (cpu)
    auto do_sum = [&]() {
        Number init = 0;
        Number sum = std::transform_reduce(
            std::execution::par, vec.begin(), vec.end(), init,
            std::plus<Number>(), [](const Number& num) { return num; });
        return sum;
    };

    Number output = 0;
    double t_main = tictoc(do_sum, output);

    print_time(t_main, "t_main");
    std::cout << "sum is " << output << std::endl;

    return 0;
}
