cmake_minimum_required(VERSION 3.10)

project(main_cpu_std_par LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TBB REQUIRED)

add_executable(main_cpu_std_par main_cpu_std_par.cpp)

# Set compiler flags for optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")
endif()

# Find and link against required libraries
find_package(Threads REQUIRED)
target_link_libraries(main_cpu_std_par PRIVATE TBB::tbb Threads::Threads)
