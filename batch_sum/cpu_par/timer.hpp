#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using MicroSec = std::chrono::microseconds;

template <typename Func, typename OutputType>
double tictoc(Func &&func, OutputType &output) {
    auto start = Clock::now();
    output = std::forward<Func>(func)();
    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<MicroSec>(end - start);
    double msec = duration.count() / 1000.0;
    return msec;
}

void print_time(double msec, std::string name = "") {
    std::string prefix{};
    if (name.length() != 0) {
        prefix = std::move(name);
    }
    std::cout << prefix << " - Execution time: " << msec << " milli seconds"
              << std::endl;
}
