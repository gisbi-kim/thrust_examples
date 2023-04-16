cmake_minimum_required(VERSION  3.22)

project(ThrustCpuGpuExample CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(THRUST_REQUIRED_SYSTEMS CPP CUDA)
set(THRUST_OPTIONAL_SYSTEMS CUDA)

find_package(Thrust REQUIRED CONFIG)

thrust_update_system_found_flags()
enable_language(CUDA)
thrust_create_target(ThrustCUDA HOST CPP DEVICE CUDA)
# thrust_create_target(ThrustCUDA HOST CPP DEVICE CPP) # if want to use cpu
add_executable(main main.cu)
target_link_libraries(main ThrustCUDA)