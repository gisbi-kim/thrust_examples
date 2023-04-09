#include <algorithm>  // generate 함수를 사용하기 위한 헤더 파일
#include <iostream>
#include <numeric>  // accumulate 함수를 사용하기 위한 헤더 파일
#include <random>
#include <vector>

#include "config.hpp"
#include "timer.hpp"

int main(void) {
    std::cout << "#####\nUsing CPU"
              << "\n######\n"
              << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9999);

    // generate random data on the host
    std::cout << "Generating " << num_samples << " random numbers ...\n"
              << std::endl;
    std::vector<Number> vec(num_samples);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });

    // compute sum on the host (cpu)
    auto do_sum = [&]() {
        Number init = 0;
        Number sum = std::accumulate(vec.begin(), vec.end(), init);
        return sum;
    };

    Number sum = 0;
    double t_main = tictoc(do_sum, sum);
    print_time(t_main, "t_main");

    std::cout << "sum is " << sum << std::endl;

    return 0;
}
